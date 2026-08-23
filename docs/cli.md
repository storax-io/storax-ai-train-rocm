# Harness CLI reference — every command as a flowchart

The harness is a set of standalone entry points, not a monolith:
`traintest/*.py` are the training-loop verbs, `tools/*.py` the
triage/ops verbs. On WSL/Linux everything runs through the venv wrapper
— `scripts/run_linux.sh <script.py> [args…]` — and on LUMI through the
container equivalents; the scripts themselves are identical everywhere.

How the verbs compose into one round of the storax loop:

```mermaid
flowchart LR
    TP["trainpack<br/>(dataset repo, versioned)"] --> TR[train.py]
    RG["gen_replay / gen_cpp_replay /<br/>gen_std_replay<br/>(model-native anchors)"] --> TR
    TR --> CK[checkpoint]
    CK --> VA[validate_ckpt.py] --> EV["oracle_eval.py<br/>(sharded, --shard I/M)"]
    EV --> ME[merge_eval.py] --> SR["strata_report.py /<br/>error_clusters.py"]
    CK --> GS["make_guard_suite.py<br/>-> oracle_eval on guard suite"]
    SR -. "coverage + drill targets" .-> TP
    CK --> HV[harvest.py] -. "winners band" .-> TP
```

## Training

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
config — single-seed comparisons are noise), `--min-free-vram` (refuses
to start on a busy card).

### Replay generators — the model-native anchors

All three share one shape; they differ in what they anchor and how the
answer is verified. **Model-native replay only** — the anchors come
from the base model being trained, never from a teacher.

```mermaid
flowchart LR
    B["BASE model answers<br/>everyday prompts"] --> V{verify}
    V -- "gen_replay: none<br/>(chat style anchor)" --> R1["replay.json (10% band)"]
    V -- "gen_cpp_replay:<br/>oracle compile+run" --> R2["cpp_replay.json<br/>(plain modern C++)"]
    V -- "gen_std_replay:<br/>gcc/g++ with the MATCHING<br/>-std= flag, per standard" --> R3["std_replay.json<br/>(C17, C++98/11/17/20 —<br/>standard named in the prompt)"]
```

### `harvest.py --model DIR --prompts P.jsonl --samples 8` — expert iteration

```mermaid
flowchart LR
    P["FRESH prompts<br/>(never the eval suite)"] --> S["best-of-N sampling<br/>(--samples, --temperature)"]
    S --> O{"oracle: compile+run"}
    O -- "a sample passes" --> W["winners.jsonl -> the next<br/>trainpack's expert band"]
    O -- "none pass" --> D[dropped]
    SH["--shard I/M"] -.-> S
```

The model teaches itself whatever it can already sample but not yet
rank first; the compiler keeps the winners.

## Evaluation

### `oracle_eval.py --model M --suite S.jsonl --out R.json` — the verdict layer

```mermaid
flowchart TB
    SU["suite JSONL {id, prompt}<br/>(--limit, --shard I/M)"] --> G["greedy generation<br/>(--backend hf|vllm, --max-new)"]
    G --> X["extract first ```-fenced block<br/>(whole output if none)"]
    X --> O{"oracle compile<br/>(--run: also execute)"}
    O -- pass --> SC[scored pass]
    O -- fail --> RP{"--repair N left?"}
    RP -- yes --> M2["repair prompt with the REAL<br/>compiler output -> regenerate"] --> O
    RP -- no --> SF[scored fail]
    SC & SF --> RJ["result JSON: rate, per-task<br/>verdicts + diagnostics"]
    TRC["--rerun-truncated PREV:<br/>redo only truncated tasks,<br/>merge by id"] -.-> G
```

Runs wide by default on LUMI: one shard per GCD, `merge_eval.py`
recombines — the eval SLO is ~10–15 min wall, never an hour.

### `merge_eval.py merged.json shard*.json`

Recomputes rates over deduped per-task results (later files win,
matching oracle_eval's own rerun-merge semantics).

### `evaluate.py` / `chat.py` / `chatprobe.py`

Phase-0 facts QA (string-keyed, pre-oracle era), interactive REPL
probing, and scripted chat probes respectively — de-risking tools, not
part of the verdict layer.

## Gates and triage

### `validate_ckpt.py` — trust no artifact you didn't verify

```mermaid
flowchart LR
    PR["producer (every segment):<br/>sha256 every shipped file +<br/>tensor spot-checks"] --> VJ["validation.json beside<br/>the checkpoint"]
    CO["consumer (next segment /<br/>eval): re-hash + compare"] --> OK{match?}
    VJ --> CO
    OK -- no --> STOP["refuse to resume<br/>(five attempts burned ~250 GPU-h<br/>on a corrupt resume source)"]
```

### `model_acid.py MODEL_DIR [--generate]` — before trusting any checkpoint on any host

```mermaid
flowchart LR
    A1["A1 tokenizer decode trap<br/>(encode identical, decode<br/>emits raw byte symbols)"] --> V2["ACID verdict line<br/>exit 0 = PASS/WARN<br/>exit 1 = any FAIL"]
    A2["A2 chat-template injection<br/>(~536-token default system<br/>prompt the model never saw)"] --> V2
    A3["A3 config/weights inventory<br/>(dtype census from safetensors,<br/>vocab agreement, provenance sha)"] --> V2
    B1["B (--generate): greedy + sampled<br/>on FRESH prompts, scoring the two<br/>convicted collapse signatures<br/>(empty-fence, cap-babble)"] --> V2
```

Self-contained on purpose (stdlib + transformers): suite hosts don't
carry the harness, and these traps live exactly in tools outside it.

### `make_guard_suite.py` — the retention guard

Builds a compiler-verified suite from `cpp_replay.json` — tasks the
base model provably answered correctly. A checkpoint that stops
compiling on them has eroded plain-C++ competence; the guard is a
**hard filter** at threshold 0.9, regardless of suite rate.

### `strata_report.py` / `error_clusters.py` — failures into generator work

```mermaid
flowchart LR
    EJ["eval JSONs<br/>(runs/**/eval/eval.json)"] --> N["normalize first_error<br/>(strip identifiers, numbers, paths)"]
    N --> CL["cluster by signature<br/>x template family"]
    CL --> MD["ranked markdown report:<br/>each cluster = one actionable —<br/>generator coverage gap OR drill target"]
    MD -.-> GEN2["next trainpack<br/>(failures are layered; each<br/>generation peels one)"]
```

## Ops

| tool | one line |
|---|---|
| `scripts/run_linux.sh` | run any traintest script under the WSL ROCm venv (closest local twin of the LUMI container) |
| `tools/estimate.py` | project measured MFU to other GPUs/model sizes |
| `tools/gpu_acceptance.py` / `node_probe.py` | is this card/node healthy enough to train on |
| `tools/cpp26_loop.py` | local dynamic rounds: train → probe-eval → add oracle-verified remedials for failing error classes → retrain |
| `tools/learning_curve.py` / `gen_report.py` | plots and run reports from metrics.jsonl |
| `tools/model_acid.py` | see above — ships to any host, no harness needed |

Dataset-side commands (`cpp26ds …`: drillgen, synthgen, harvest-packages,
trainpack, …) live in the dataset repo — flowcharts in
[storax-dataset-cpp26 docs/cli.md](../../storax-dataset-cpp26/docs/cli.md).
