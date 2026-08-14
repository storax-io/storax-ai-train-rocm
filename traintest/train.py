"""Full fine-tune of SmolLM3-3B on synthetic unknown facts, on ROCm.

Memory recipe for 3B params in 16 GiB VRAM (RX 7800 XT):
  bf16 weights (6.2 GiB) + bf16 grads (6.2 GiB) + Adafactor factored states
  (~MBs, vs 24 GiB for fp32 AdamW) + gradient checkpointing + short seqs.
AdamW does not fit — do not "upgrade" the optimizer without re-budgeting.

Writes run artifacts to --out: metrics.jsonl (per-step), result.json
(summary incl. tokens/sec and peak VRAM), and the trained model.
"""
import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import warnings

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

import facts
import hfcompat

# Vendor peak dense bf16 (TFLOPS) for MFU. RDNA3 dual-issue peak — real
# kernels rarely exceed ~30-40% of it even on mature stacks.
PEAK_BF16_TFLOPS = {
    "AMD Radeon RX 7800 XT": 74.65,
}


def build_samples(tok, seq_len, only_think=False, system=None, data=facts):
    """Every sample is exactly seq_len tokens so all steps share one
    (batch, seq_len) shape — ragged shapes make ROCm re-autotune GEMMs on
    nearly every step (measured: 0.7s vs 12s for the same work).

    Full-loss text (article chunks + fact statements) is packed into dense
    blocks separated by EOS; chat-format QA (loss on answer only) is padded
    to seq_len."""
    samples = []
    stream = []
    for text in ([] if only_think
                 else data.article_texts() + data.training_texts()):
        stream += tok(text, add_special_tokens=False).input_ids
        stream.append(tok.eos_token_id)
    for i in range(0, len(stream) - seq_len + 1, seq_len):
        ids = torch.tensor(stream[i : i + seq_len])
        samples.append((ids, ids.clone(), torch.ones(seq_len, dtype=torch.long)))

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    qa_dropped = 0

    def chat_sample(prompt, completion, thinking):
        """Chat-format sample, loss on completion only, padded to seq_len.
        Returns None if it doesn't fit (caller counts drops)."""
        prompt_ids = hfcompat.chat_prompt_ids(
            tok, [{"role": "user", "content": prompt}], thinking=thinking,
            system=system)
        ans_ids = tok(completion + tok.eos_token, return_tensors="pt",
                      add_special_tokens=False).input_ids[0]
        if len(prompt_ids) + len(ans_ids) > seq_len:
            return None
        ids = torch.cat([prompt_ids, ans_ids])
        labels = ids.clone()
        labels[: len(prompt_ids)] = -100
        attn = torch.ones(len(ids), dtype=torch.long)
        fill = seq_len - len(ids)
        if fill:
            ids = torch.cat([ids, torch.full((fill,), pad_id)])
            labels = torch.cat([labels, torch.full((fill,), -100)])
            attn = torch.cat([attn, torch.zeros(fill, dtype=torch.long)])
        return (ids, labels, attn)

    # QA pairs carry loss on only ~10 answer tokens each; x3 so the eval
    # format gets meaningful gradient share vs the article blocks.
    # chat_sample DROPS oversized samples — the old truncate-then-mask
    # path produced all(-100)-label samples: NaN loss, zero gradient,
    # silently untrained QA (found on Ministral, whose template injects a
    # ~200-token default system prompt).
    for q, a in ([] if only_think else data.training_qa_pairs()):
        s = chat_sample(q, a, False)
        if s is None:
            qa_dropped += 1
        else:
            samples += [s] * 3
    if qa_dropped:
        print(f"qa: {qa_dropped} samples over seq_len, dropped", flush=True)

    # Thinking-mode QA: constructed traces over ground truth — ONLY for
    # models with a dedicated <think> token. For anything else the tags
    # are plain text, and the model learns to spray "<think> Recalling
    # the facts: ..." boilerplate into ordinary answers (observed on
    # Ministral: adjacent-knowledge answers polluted with it).
    has_think_token = len(tok("<think>", add_special_tokens=False).input_ids) == 1
    if not has_think_token and not only_think:
        print("think-qa: skipped (tokenizer has no <think> token)", flush=True)
    probe = tok.apply_chat_template(
        [{"role": "user", "content": "x"}], add_generation_prompt=True,
        enable_thinking=True, tokenize=False)
    open_tag = "" if probe.rstrip().endswith("<think>") else "<think>\n"
    dropped = 0
    for q, trace, ans in (data.think_qa_pairs() if has_think_token else []):
        s = chat_sample(q, f"{open_tag}{trace}\n</think>\n{ans}", True)
        if s is None:
            dropped += 1
        else:
            samples.append(s)
    if dropped:
        print(f"think-qa: {dropped} samples over seq_len, dropped", flush=True)

    # Replay anchors: base-model answers to general prompts (both modes).
    # Keeps chat style, world knowledge, and the thinking format itself
    # from drifting toward the domain corpus.
    here = Path(__file__).resolve().parent
    # replay_think.json intentionally absent: natural SmolLM3 traces run
    # ~650 tokens (measured) and cannot fit seq_len — thinking style is
    # trained solely via the short constructed traces above.
    for fname, thinking in (("replay.json", False), ("replay_think.json", True)):
        # Local (model-specific, freshly generated) wins over the repo's
        # data/ copy. Missing replay is loud: mistral1 trained without
        # anchors because only the Windows staging dir had the file, and
        # adjacent knowledge collapsed 87.5% -> 12.5%.
        rdir = os.environ.get("TRAINTEST_REPLAY_DIR")
        cands = ([Path(rdir) / fname] if rdir else []) + \
                [here / fname, here.parent / "data" / fname]
        f = next((c for c in cands
                  if c.exists()), None)
        if f is None:
            if fname == "replay.json":
                print("replay: replay.json NOT FOUND — training unanchored, "
                      "style/knowledge drift likely", flush=True)
            continue
        n = 0
        for r in json.loads(f.read_text(encoding="utf-8")):
            s = chat_sample(r["prompt"], r["answer"], thinking)
            if s is not None:
                samples.append(s)
                n += 1
        print(f"replay: {n} anchor samples from {fname}", flush=True)
    return samples


def collate(batch):
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    attn = torch.stack([b[2] for b in batch])
    return input_ids, labels, attn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM3-3B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = no cap")
    ap.add_argument("--lr", type=float, default=3e-5)
    # VRAM headroom rules this card, not batch size: batch 4 paged from
    # step 1 (full3); batch 2 at seq 320 decayed into paging mid-run as
    # fragmentation + desktop usage grew (full7). batch 1 + accum 8 keeps
    # the same effective batch with maximum slack; on a headless node,
    # raise batch instead.
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0,
                help="shuffle seed (multi-seed variance runs)")
    ap.add_argument("--seq-len", type=int, default=320)  # room for think traces
    ap.add_argument("--attn", default="sdpa",
                    choices=["sdpa", "eager", "flash_attention_2"])
    # flash_attention_2: LUMI container (CK flash-attn on gfx90a); not
    # available in the Windows ROCm preview stack.
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile via Inductor/Triton (experimental on Windows ROCm)")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--only-think", action="store_true",
                    help="continue-training: think-QA + replay anchors only "
                         "(use with --model <trained dir>)")
    ap.add_argument("--min-free-vram", type=float, default=6.0,
                    help="GiB of free VRAM required at launch (model load "
                         "happens before this check; grads/activations after)")
    ap.add_argument("--dist-backend", default="",
                    help="override process-group backend (default: nccl "
                         "with CUDA/ROCm, gloo on CPU)")
    ap.add_argument("--grad-sync", default="manual",
                    choices=["manual", "ddp"],
                    help="manual: plain module + in-place grad allreduce at "
                         "accumulation boundaries (no extra memory — DDP's "
                         "reducer buckets + engine grads need 2x gradient "
                         "memory and cannot fit 14B on a 64GB GCD); ddp: "
                         "wrapper with overlapped comm, for models that fit")
    ap.add_argument("--data", default="facts",
                    choices=["facts", "cpp26", "cpp26ds"],
                    help="training data provider module")
    ap.add_argument("--system", default=None,
                    help="short system-prompt override for all chat samples "
                         "(use the same value in evaluate.py)")
    ap.add_argument("--freeze", default="",
                    help="comma-separated param-name substrings to freeze, "
                         "e.g. vision_tower,multi_modal_projector,embed_tokens,lm_head")
    args = ap.parse_args()

    # Multi-node/multi-GPU via torchrun: RANK/WORLD_SIZE/LOCAL_RANK env.
    # Backend: nccl (= RCCL on ROCm/LUMI) with GPUs, gloo for CPU
    # simulation — the simulated path exercises the same DDP mechanics
    # (sharding, no_sync accumulation, rank-0 artifacts) as a real node.
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available()

    # NUMA affinity: pin this rank's threads to its GPU's closest cores.
    # TRAINTEST_CORE_MAP="49-55,57-63,..." — one range per local rank,
    # indexed by LOCAL_RANK (LUMI-G: each GCD has a 7-core L3 group;
    # verify the mapping on-node with rocm-smi --showtoponuma first).
    # Children (OpenMP, dataloader) inherit the affinity.
    core_map = os.environ.get("TRAINTEST_CORE_MAP", "")
    if core_map:
        ranges = [r.strip() for r in core_map.split(",")]
        if local_rank < len(ranges):
            lo, _, hi = ranges[local_rank].partition("-")
            cores = set(range(int(lo), int(hi or lo) + 1))
            os.sched_setaffinity(0, cores)
            os.environ.setdefault("OMP_NUM_THREADS", str(len(cores)))
            if local_rank == 0:
                print(f"affinity: rank0 -> cores {sorted(cores)} "
                      f"(map has {len(ranges)} entries)", flush=True)
    if world > 1:
        dist.init_process_group(
            args.dist_backend or ("nccl" if use_cuda else "gloo"))
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)  # current-device APIs (allocator
        # stats, context creation) must target this rank's GCD, not GPU 0
    is_main = rank == 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics_f = ((out / "metrics.jsonl").open("w") if is_main
                 else open(os.devnull, "w"))

    tok = hfcompat.load_tokenizer(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    model = hfcompat.load_causal_model(args.model, torch.bfloat16, args.attn)
    model.to(device)
    # non-reentrant checkpointing is required under DDP
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = False
    model.train()

    if args.freeze:
        frozen = 0
        keys = [k.strip() for k in args.freeze.split(",") if k.strip()]
        for name, p in model.named_parameters():
            if any(k in name for k in keys):
                p.requires_grad_(False)
                frozen += p.numel()
        if is_main:
            print(f"freeze: {frozen/1e9:.2f}B params frozen ({args.freeze})",
                  flush=True)

    if world > 1 and args.grad_sync == "ddp":
        # gradient_as_bucket_view halves steady-state grad memory, but the
        # reducer's eager buckets + engine-allocated grads still peak at 2x
        # gradient size (verified: single-rank probe backward-2 peak, and
        # 14B OOM on 64GB GCDs in LUMI jobs 21136222/21136902) — use the
        # default manual sync for anything that doesn't obviously fit.
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if use_cuda else None,
            find_unused_parameters=False, gradient_as_bucket_view=True)

    if args.compile:
        model = torch.compile(model)

    # transformers Adafactor with scale_parameter=False applies --lr as an
    # absolute step size. torch.optim.Adafactor multiplies lr by parameter
    # RMS (~0.02), which silently turned 1e-5 into ~2e-7 and produced a run
    # that trained nothing (full1: 29%->33%). Do not switch back casually.
    from transformers.optimization import Adafactor
    opt = Adafactor([p for p in model.parameters() if p.requires_grad],
                    lr=args.lr, scale_parameter=False, relative_step=False,
                    warmup_init=False)

    if args.data == "cpp26":
        import cpp26data as data_mod
    elif args.data == "cpp26ds":
        import cpp26dsdata as data_mod
    else:
        data_mod = facts
    samples = build_samples(tok, args.seq_len, only_think=args.only_think,
                            system=args.system, data=data_mod)
    g = torch.Generator().manual_seed(args.seed)

    # Constant 3e-5 to the last step tipped full5 into mode collapse
    # (unrelated questions answered with verbatim training sentences;
    # final loss 0.44 vs healthy full4's 0.75). Linear decay to 0 ends the
    # run gently; 20-step warmup avoids the first-step shock.
    shard_len = len(samples) // world
    total_steps = args.max_steps or args.epochs * (shard_len // args.batch)
    warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*optimizer.step.*")
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / 20, max(0.0, 1 - s / max(total_steps, 1))))

    if use_cuda:
        free_b, total_b = torch.cuda.mem_get_info()
        if is_main:
            print(f"vram: {free_b / 2**30:.1f} GiB free of "
                  f"{total_b / 2**30:.1f} before training "
                  f"(desktop shares this GPU)", flush=True)
        # Refuse doomed runs instead of crashing into them: the desktop
        # (DWM has been observed leaking to 18+ GiB) can leave too little
        # VRAM, and the failure then surfaces as confusing HIP errors or
        # silent paging mid-run. Override with --min-free-vram 0.
        if free_b / 2**30 < args.min_free_vram:
            raise SystemExit(
                f"ABORT: only {free_b / 2**30:.1f} GiB free VRAM "
                f"(< {args.min_free_vram} required). Close GPU-heavy apps "
                f"or restart dwm.exe, then retry.")
        torch.cuda.reset_peak_memory_stats()
    step = 0
    tokens_done = 0
    step_rates = []  # tok/s per step, for steady-state (warmup-free) stats
    t_start = time.perf_counter()
    stop = False
    for epoch in range(args.epochs):
        # same seed on every rank -> identical permutation; each rank
        # takes a disjoint stride so the union covers the epoch exactly.
        perm = torch.randperm(len(samples), generator=g).tolist()[rank::world]
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            batch = [samples[j] for j in perm[i : i + args.batch]]
            input_ids, labels, attn = collate(batch)
            input_ids, labels, attn = (input_ids.to(device),
                                       labels.to(device), attn.to(device))

            t0 = time.perf_counter()
            will_sync = (step + 1) % args.accum == 0
            # skip gradient allreduce on non-boundary accumulation steps
            # (no_sync exists only on the DDP wrapper; the manual path
            # simply doesn't communicate until the boundary)
            ctx = (model.no_sync()
                   if world > 1 and args.grad_sync == "ddp" and not will_sync
                   else contextlib.nullcontext())
            try:
                with ctx:
                    loss = model(input_ids=input_ids, attention_mask=attn,
                                 labels=labels).loss
                    (loss / args.accum).backward()
            except torch.OutOfMemoryError:
                # the allocator table names where the bytes sit — worth more
                # than the traceback on a machine we can't interactively probe
                print(f"rank{rank} OOM at step {step + 1}; allocator state:\n"
                      f"{torch.cuda.memory_summary(device=device, abbreviated=True)}",
                      flush=True)
                raise
            if will_sync and world > 1 and args.grad_sync == "manual":
                # in-place allreduce on param.grad: zero extra memory,
                # amortized over the accumulation window
                grads = [p.grad for p in model.parameters()
                         if p.grad is not None]
                works = [torch.distributed.all_reduce(g, async_op=True)
                         for g in grads]
                for w in works:
                    w.wait()
                torch._foreach_mul_(grads, 1.0 / world)
                # these locals otherwise persist to the NEXT boundary and
                # pin all 22.5 GiB of freed grads through the following
                # backward — double-grad OOM (LUMI job 21138910)
                del grads, works
            if will_sync:
                opt.step()
                opt.zero_grad(set_to_none=True)
            sched.step()
            if use_cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            step += 1
            tokens_done += int(attn.sum().item())
            rec = {"step": step, "epoch": epoch, "loss": round(loss.item(), 4),
                   "step_s": round(dt, 3),
                   "tok_per_s": round(int(attn.sum().item()) / dt, 1)}
            if use_cuda:
                # allocated now vs peak-so-far: a growing 'mem' with flat
                # 'peak' distinguishes a leak from a first-step transient
                rec["mem_gib"] = round(torch.cuda.memory_allocated() / 2**30, 2)
                rec["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            # MFU accounting uses computed positions: padded QA slots burn
            # the same FLOPs as real tokens even though loss ignores them.
            step_rates.append(input_ids.numel() / dt)
            metrics_f.write(json.dumps(rec) + "\n")
            metrics_f.flush()
            if step % 10 == 0 and is_main:
                print(json.dumps(rec), flush=True)
            if args.max_steps and step >= args.max_steps:
                stop = True
                break
        if stop:
            break

    wall = time.perf_counter() - t_start
    n_params = sum(p.numel() for p in model.parameters())
    # Training FLOPs/token: fwd 2N + bwd 4N + checkpoint recompute fwd 2N.
    flops_per_token = 8 * n_params
    dev_name = torch.cuda.get_device_name(0) if use_cuda else "cpu"
    peak_tflops = PEAK_BF16_TFLOPS.get(dev_name)
    # Steady-state rate: median over post-warmup steps — first steps pay
    # one-off kernel autotune costs and would understate the hardware.
    # step_rates are computed-position rates (see above), so this is the
    # hardware utilization figure, not training-progress tokens.
    steady = sorted(step_rates[3:]) if len(step_rates) > 6 else sorted(step_rates)
    tok_s = steady[len(steady) // 2] if steady else 0.0
    summary = {
        "steps": step,
        "final_loss": rec["loss"] if step else None,
        "wall_s": round(wall, 1),
        "tokens": tokens_done,
        "tok_per_s_avg": round(tokens_done / wall, 1),
        "computed_tok_per_s_steady": round(tok_s, 1),
        "peak_vram_gib": (round(torch.cuda.max_memory_allocated() / 2**30, 2)
                          if use_cuda else None),
        "world_size": world,
        "global_tok_per_s_avg": round(tokens_done * world / wall, 1),
        "params_b": round(n_params / 1e9, 3),
        "achieved_tflops": round(tok_s * flops_per_token / 1e12, 2),
        "peak_bf16_tflops": peak_tflops,
        "mfu": (round(tok_s * flops_per_token / (peak_tflops * 1e12), 4)
                if peak_tflops else None),
        "attn": args.attn,
        "compiled": args.compile,
        "optimizer": "adafactor",
        "lr": args.lr,
        "seq_len": args.seq_len,
        "batch": args.batch,
        "device": dev_name,
    }

    if world > 1:
        dist.barrier()
    if args.save_model and is_main:
        target = model._orig_mod if hasattr(model, "_orig_mod") else model
        target = target.module if hasattr(target, "module") else target
        target.save_pretrained(out / "model", safe_serialization=True)
        tok.save_pretrained(out / "model")
        summary["model_dir"] = str(out / "model")

    if is_main:
        (out / "result.json").write_text(json.dumps(summary, indent=2))
        print("RESULT " + json.dumps(summary), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
