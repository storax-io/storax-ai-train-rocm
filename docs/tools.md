# Harness tools — every script as a flowchart

The harness is a set of standalone entry points, not a monolith:
`traintest/*.py` are the training-loop verbs, `tools/*.py` the
gate/triage/ops verbs, `scripts/*.sh` the runners. On LUMI these are
orchestrated by the `sc` campaign CLI ([usage reference](cli.md);
closed source) as plain Slurm jobs. This page documents the tools
themselves — they run identically under the local runners and inside
the container.

How the tools compose into one round of the storax loop:

```mermaid
flowchart LR
    TP["trainpack<br/>(dataset repo, versioned)"] --> TR[train.py]
    RG["gen_replay / gen_cpp_replay /<br/>gen_std_replay<br/>(model-native anchors)"] --> TR
    SH["stage-harness.sh<br/>(committed-tree snapshot,<br/>HARNESS_COMMIT stamped)"] -.-> TR
    TR --> CK[checkpoint]
    CK --> VA[validate_ckpt.py] --> EV["oracle_eval.py<br/>(sharded, --shard I/M)"]
    EV --> ME[merge_eval.py] --> RPT["gen_report.py /<br/>strata_report.py / error_clusters.py"]
    CK --> GS["make_guard_suite.py<br/>-> oracle_eval on guard suite"]
    RPT -. "coverage + drill targets" .-> TP
    CK --> HV[harvest.py] -. "winners band" .-> TP
```

Local runners: `scripts/run_linux.sh <script.py>` (WSL ROCm venv — the
local twin of the LUMI container), `scripts/run_win.sh` (Windows venv,
syncs to C: staging first), `scripts/stage-harness.sh` (committed-tree
snapshot via `git archive`, HARNESS_COMMIT stamped — a dirty tree
cannot leak).

### `train.py --data cpp26ds --out DIR` — one round

```mermaid
flowchart TB
    IN["--model base + --data provider<br/>(cpp26ds: bands, caps, MIX —<br/>see architecture.md corpus-intake)"] --> MB["fixed batch shapes<br/>(--batch/--accum/--seq-len;<br/>ragged batches cost 16x on ROCm)"]
    MB --> ST["bf16 + Adafactor + grad ckpt<br/>warmup + linear LR decay to 0"]
    ST --> SYNC{"multi-GPU?"}
    SYNC -- "--grad-sync manual" --> AR["in-place all-reduce at<br/>accumulation boundaries<br/>(DDP cannot fit a 14B on 64GB)"]
    ST --> LG["metrics.jsonl per step"]
    LG --> AB{"sustained loss <= 1e-3?"}
    AB -- yes --> KILL["scripted ABORT —<br/>memorization depth breaks<br/>guards and repair"]
    ST --> OUT["--out: model (--save-model),<br/>result.json (tok/s, peak VRAM),<br/>optional midsaves + resumable state"]
```

Key flags: `--total-steps` (LR schedule truth for multi-segment runs),
`--resume-state/--save-state`, `--freeze`, `--seed` (≥3 seeds per
config), `--min-free-vram` (refuses a busy card). **Data providers**
(`--data` plugins, not commands): `cpp26dsdata.py` (bands/caps/MIX),
`cpp26data.py` (earlier corpus provider), `facts.py` (phase-0);
`hfcompat.py` is the transformers 4/5 chat-template shim under all of
them.

### Replay generators — the model-native anchors

```mermaid
flowchart LR
    B["BASE model answers<br/>everyday prompts"] --> V{verify}
    V -- "gen_replay: none<br/>(chat style anchor)" --> R1["replay.json (10% band)"]
    V -- "gen_cpp_replay:<br/>oracle compile+run" --> R2["cpp_replay.json<br/>(plain modern C++)"]
    V -- "gen_std_replay:<br/>gcc/g++ with the MATCHING<br/>-std= flag, per standard" --> R3["std_replay.json<br/>(C17, C++98/11/17/20 —<br/>standard named in the prompt)"]
```

**Model-native replay only** — anchors come from the model being
trained, never a teacher. On LUMI the first of these runs at scale as the retention-band job.

### `oracle_eval.py --model M --suite S.jsonl` — the verdict layer

```mermaid
flowchart TB
    SU["suite JSONL {id, prompt}<br/>(--limit, --shard I/M)"] --> G["greedy generation<br/>(--backend hf|vllm, --max-new)"]
    G --> X["extract first ```-fenced block<br/>(whole output if none)"]
    X --> O{"oracle compile<br/>(--run: also execute)"}
    O -- pass --> SC2[scored pass]
    O -- fail --> RP{"--repair N left?"}
    RP -- yes --> M2["repair prompt with the REAL<br/>compiler output -> regenerate"] --> O
    RP -- no --> SF[scored fail]
    SC2 & SF --> RJ2["result JSON: rate, per-task<br/>verdicts + diagnostics"]
    TRC["--rerun-truncated PREV:<br/>redo only truncated tasks,<br/>merge by id"] -.-> G
```

`merge_eval.py merged.json shard*.json` recombines shards (later files
win, matching oracle_eval's own rerun-merge semantics). This is the judge behind every sharded eval on LUMI.

### `harvest.py` — expert iteration

Best-of-N sampling on fresh prompts; the oracle keeps winners →
`winners.jsonl` → the next trainpack's expert band. The model teaches
itself whatever it can already sample but not yet rank first.

### Gates

```mermaid
flowchart LR
    subgraph vc["tools/validate_ckpt.py"]
        PR["producer: sha256 every shipped<br/>file + tensor spot-checks<br/>-> validation.json"] --> CO["consumer: re-hash before<br/>resume/eval; mismatch = refuse<br/>(five attempts burned ~250 GPU-h<br/>on a corrupt resume source)"]
    end
    subgraph ma["tools/model_acid.py (any host, stdlib+transformers)"]
        A1["A1 tokenizer decode trap"] & A2["A2 chat-template injection"] & A3["A3 config/weights inventory"] & B1["B generation battery<br/>(collapse signatures)"] --> V2["ACID verdict, exit 1 = FAIL"]
    end
```

`tools/make_guard_suite.py` builds the retention-guard suite from
`cpp_replay.json` — tasks the base model provably answered; the guard
is a **hard filter** at 0.9 regardless of suite rate.

### Reports and triage

```mermaid
flowchart LR
    EJ["eval JSONs + metrics.jsonl"] --> GR["gen_report.py — the DECISION<br/>CONTRACT: seeds, mean/min-max/sigma,<br/>guard-min, first-shot vs repaired"]
    EJ --> LC["learning_curve.py — one row<br/>per generation, campaign headline"]
    EJ --> UP["user_pain.py — the eval as 128<br/>users: wall-clock felt per verdict"]
    EJ --> N["strata_report.py / error_clusters.py<br/>normalized error signature x family<br/>-> ranked actionables"]
    N -.-> GEN2["next trainpack (each generation<br/>peels one failure layer)"]
    LOOP["tools/cpp26_loop.py — local dynamic<br/>rounds: train -> probe -> add verified<br/>remedials for the error CLASS -> retrain"] ~~~ EJ
```

### `tools/quantize.py` — checkpoint to serving artifacts

Quantize-only by design: a quant is judged by the eval battery, never
by its maker.

**Cost:** CPU-minutes (convert + requantize; ~10–20 min for a 14B).
**Touches:** the --out dir + vendored tools/llama.cpp (pinned tag).

```mermaid
flowchart LR
    CK2["HF checkpoint dir"] --> CV["convert_hf_to_gguf -> f16 GGUF"]
    CAL["calibration text sampled from<br/>OUR trainpack (--calib-pack) —<br/>the deployment distribution,<br/>not wikitext"] --> IM["llama-imatrix<br/>(default; --no-imatrix opts out)"]
    CV --> IM
    IM --> QQ
    LL["llama.cpp vendored at pinned tag,<br/>built once: HIP if hipcc exists,<br/>CPU otherwise (quantization is<br/>CPU-bound either way)"] --> CV
    QQ["llama-quantize --imatrix per<br/>requested quant (q4_k_m, q5_k_m, q8_0)"]
    QQ --> Rep["quantize-report.json:<br/>sizes + sha256 per artifact"]
    QQ -.-> JUDGE2["judging: sc bench the quant<br/>through the same oracle —<br/>NOT this tool's job"]
```

### Probes and environment (run before anything expensive)

| tool | one line |
|---|---|
| `traintest/env_probe.py` | ROCm/PyTorch env probe, one JSON, never raises — failures are diagnoses |
| `traintest/triton_probe.py` | compile+run real Triton kernels, numerics vs eager — the Instinct-pathfinding test |
| `traintest/impcheck.py` | surface the real exception behind transformers' lazy-import wrapper |
| `traintest/download.py` | pre-fetch a model into the HF cache |
| `traintest/chat.py` | interactive GPU REPL (`/think`, `/temp`, `/clear`) |
| `traintest/chatprobe.py` | scripted chat probes: general-chat drift, base vs tuned |
| `traintest/thinkprobe.py` | thinking-mode template mechanics + natural trace length |
| `tools/gpu_acceptance.py` | GPU acceptance: VRAM pattern-fill integrity + sustained bf16 burst |
| `tools/node_probe.py` | per-node health: alloc+GEMM+sync every GCD + filesystem touch — a wedged GPU fails in seconds, not mid-round |
| `tools/estimate.py` | project measured MFU to other GPUs/model sizes |
| `tools/quantize.py` | checkpoint -> GGUF quants for serving (llama.cpp vendored+pinned, HIP when hipcc exists, else CPU); sizes+sha256 report; judging a quant stays with the eval battery |

### Phase-0 lineage (kept for the de-risking story)

`evaluate.py` (string-keyed facts QA), `tools/build_dataset.py`
(Wikipedia → facts corpus), `tools/build_cpp26_corpus.py` (first
oracle-verified C++26 corpus builder; superseded by the dataset repo).

---

Dataset-side commands (`cpp26ds …`: drillgen, synthgen,
harvest-packages, trainpack, …) live in the dataset repo — flowcharts
in [storax-dataset-cpp26 docs/cli.md](https://github.com/storax-io/storax-dataset-cpp26/blob/main/docs/cli.md).
