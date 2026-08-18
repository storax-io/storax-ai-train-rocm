"""Checkpoint validation gate (2026-08-18: five segment attempts burned
~250 GPU-h before anyone checked whether the resume SOURCE was sound —
and a CG-stuck node with a failing Lustre write path had produced it).

Producer: every segment runs this on its shipped artifacts and writes
validation.json (per-file sha256 + tensor spot-checks). Consumer: the
template REFUSES to resume from a directory without a valid stamp.

Checks:
  * every *.safetensors: header parses, K sampled tensors per shard are
    all-finite with nonzero spread (catches zero-page/garbage regions)
  * train_state.pt: unpickles, carries step/optimizer keys
  * sha256 of every file -> validation.json (transport-corruption proof)

    python3 tools/validate_ckpt.py SEG_DIR [--samples 8]
Exit 0 + validation.json on pass; exit 1 with the named failure.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def check_shard(path: Path, k: int):
    from safetensors import safe_open
    with safe_open(str(path), framework="pt", device="cpu") as f:
        names = list(f.keys())
        if not names:
            return f"{path.name}: no tensors"
        step = max(1, len(names) // k)
        for name in names[::step][:k]:
            t = f.get_tensor(name)
            tf = t.float()
            if not bool(tf.isfinite().all()):
                return f"{path.name}:{name}: non-finite values"
            if t.numel() > 4 and float(tf.std()) == 0.0:
                return f"{path.name}:{name}: zero spread (zeroed region?)"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seg_dir")
    ap.add_argument("--samples", type=int, default=8)
    args = ap.parse_args()
    seg = Path(args.seg_dir)
    model = seg / "model"
    report = {"files": {}, "checks": []}

    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        print(f"VALIDATE FAIL: no safetensors under {model}")
        return 1
    for p in [*shards, *model.glob("*.json"), seg / "train_state.pt"]:
        if not p.exists():
            continue
        report["files"][str(p.relative_to(seg))] = hashlib.sha256(
            p.read_bytes()).hexdigest()
    for shard in shards:
        err = check_shard(shard, args.samples)
        if err:
            print(f"VALIDATE FAIL: {err}")
            return 1
        report["checks"].append(f"{shard.name}: {args.samples} tensors finite+spread OK")

    state = seg / "train_state.pt"
    if state.exists():
        import torch
        st = torch.load(str(state), map_location="cpu", weights_only=False)
        if "optimizer" not in st or "step" not in st:
            print("VALIDATE FAIL: train_state.pt missing optimizer/step")
            return 1
        report["checks"].append(f"train_state.pt: step {st['step']} OK")

    (seg / "validation.json").write_text(json.dumps(report, indent=1))
    print(f"VALIDATED: {seg} ({len(shards)} shards, "
          f"{len(report['files'])} files hashed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
