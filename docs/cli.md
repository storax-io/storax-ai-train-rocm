# `sc` — the campaign CLI (usage reference)

`sc` is a **front end to Slurm**: it owns no runtime of its own — one
command turns an intent ("run this generation", "judge every
checkpoint", "train 1B tokens as a segment chain") into the right
Slurm job chain, and all causality lives in Slurm dependencies.
**Closed source, open usage**: the implementation is private; this page
documents the command surface, one flowchart per verb.

Design rules that hold everywhere: everything submitted is a plain
Slurm job — chains via `afterok`, failure janitors via `afternotok`,
**no daemons, no cron**. Errors pause the line; hardware faults
self-heal; nothing burns to a time cap. Every submit verb is
**idempotent** — re-running the same command resumes or converges
instead of duplicating. The jobs execute the open-source harness tools
in this repo ([tools.md](tools.md)).

Every submit verb passes the same gate first:

```mermaid
flowchart LR
    V[any submit verb] --> PZ{"line paused?"}
    PZ -- yes --> DIE["refuse: review,<br/>then sc resume"]
    PZ -- no --> WX{"fresh weather<br/>verdict bad?"}
    WX -- "storage dark<br/>< 15 min ago" --> DIE2["WEATHER HOLD —<br/>billed nodes never<br/>probe out a storm"]
    WX -- clear/stale --> Q[("Slurm queue")]
```

## Launching training

### `sc preflight`

Zero-cost readiness check. Verifies the staged code and data on the login node before anything is allowed to queue — a failed preflight blocks every launch verb.

**Cost:** none (login-side). **Touches:** nothing — read-only verification.

```mermaid
flowchart LR
    P[sc preflight] --> C["verify staged code + data<br/>against the sync manifest<br/>(login-side, zero cost)"]
    C -- clean --> OK[ready to submit]
    C -- mismatch --> X["nonzero exit —<br/>nothing may launch"]
```

### `sc gen MANIFEST`

Launches a whole generation from a manifest file: one training round per row, each with its failure janitor and a chained wide eval, so the verdicts arrive while later rounds are still training.

**Cost:** one training round per manifest row (nodes × steps as specified) + one wide-eval node (~10 min) per row; janitors are ~free CPU jobs. **Touches:** runs/GEN/*, logs.

```mermaid
flowchart LR
    MF["manifest rows:<br/>gen round mix drill seed steps"] --> PF[preflight]
    PF --> RJ["one round job per row"]
    RJ -- afternotok --> JN["janitor<br/>(node faults only)"]
    RJ -- afterok --> EV["WIDE eval: own node,<br/>8-way sharded, ~10 min —<br/>verdicts overlap later<br/>rounds' training"]
    EV -- afternotok --> EJ[evaljanitor]
    RJ & EV --> OUT["generation queued —<br/>safe to log out"]
```

### `sc rounds GEN ROUND MIX DRILL STEPS SEED…`

Launches one recipe across a list of seeds — the seed-replication form of `gen` (≥3 seeds per config; single-seed comparisons are noise).

**Cost:** one round + one eval per seed. **Touches:** runs/GEN/ROUND-sSEED.

```mermaid
flowchart LR
    A["one recipe,<br/>a list of seeds"] --> L{"per seed"}
    L --> RJ["round job + janitor"]
    RJ -- afterok --> EV["chained eval<br/>(long generation cap)"]
```

### `sc consolidate NAME MIX DRILL SEED [MTOK]`

Trains a large token budget as an idempotent chain of Slurm segments
sharing one LR schedule, widening across a node ladder, with concurrent
per-segment evals and automatic plateau cutoff. Re-running the same
command is the recovery procedure — completed segments are detected on
disk and skipped, so a re-run only ever spends the *remaining* token
budget, and the plateau cutoff can shrink even that.

**Cost:** bounded by the UNSPENT part of MTOK (done segments skip; plateau cutoff cancels the pending tail) + one eval node per segment + one spare node per segment. **Touches:** runs/consol/NAME/seg*.

```mermaid
flowchart TB
    LAD["width ladder (e.g. 8->16->32->64 nodes,<br/>custom --ladder n:mtok,…)<br/>ONE LR schedule across segments"] --> SEGS{"per segment:<br/>state already<br/>on disk?"}
    SEGS -- done --> SKIP["skipped — re-running the same<br/>command IS the recovery"]
    SEGS -- mid-checkpoint --> MID["resume from the<br/>segment's midpoint"]
    SEGS -- missing --> SB["submit segment, afterok on the<br/>previous (+1 spare node: a sick<br/>node idles, doesn't kill the run)"]
    SB --> CEVAL["segment evals run CONCURRENTLY<br/>on their own nodes — the chain<br/>never waits on a verdict"]
    CEVAL --> CONV["converge director per segment:<br/>gain < --improve-min -> cancel the<br/>still-PENDING tail. The plateau is<br/>the cutoff, not the token plan.<br/>A missing eval cancels NOTHING."]
    CTRL["--controlled: submit only the next<br/>segment; each director computes<br/>family weights from its eval and<br/>extends the chain itself"] -.-> SB
    FORK["--from DIR: fork any validated<br/>checkpoint into a parallel branch"] -.-> SB
    TAIL["chain fully trained but final eval<br/>missing (e.g. a timeout took the<br/>dependent eval down)? re-running<br/>submits exactly the missing eval"] -.-> CEVAL
```

### `sc campaign PLAN.json [--continue|--reset]`

Runs an unattended multi-stage arc from a JSON plan: each stage's terminal jobs chain a director job that gates on artifacts and submits the next stage. A failed gate pauses the line with the reason.

**Cost:** the sum of its stages, one stage at a time; directors are ~free 10-min CPU jobs. **Touches:** campaign state/log + whatever each stage touches.

```mermaid
flowchart TB
    PL["plan: JSON stage list<br/>(gen | consolidate | replay | sweep),<br/>each with an artifact gate"] --> KEY["state keyed to the PLAN —<br/>a new plan starts at stage 0;<br/>a stale --continue is ignored"]
    KEY --> G0{"--continue: gate the<br/>finished stage — any<br/>(rate, guard, medtok)<br/>candidate passes?"}
    G0 -- fail --> PAU["pause with the reason"]
    G0 -- pass --> ST["submit stage i"]
    ST --> DIR["director: tiny CPU job on<br/>afterany of the stage's terminal<br/>jobs -> sc campaign --continue"]
    DIR --> KEY
    RST["--reset: refused while the plan's<br/>own jobs are still queued/running"] -.-> KEY
```

### `sc reconcile [--plan P]`

Converges reality toward the plan: observes what exists (run tree,
queue) versus what should, and submits **exactly what is missing —
never what already exists**. The recovery verb for PLAN-FILE campaigns
— it reads the campaign state, so it will faithfully chase whatever
plan that state names. A consolidate launched directly (no plan file)
recovers by re-running its own command, which is equally idempotent;
retire stale campaign state first or reconcile will resurrect the old
plan. Reconcile itself is free (login-side,
read-only); any GPU cost you see afterwards is the *unfinished
remainder of the original plan* resuming, not work being redone.

**Cost:** free in itself (login-side, read-only); anything it submits is the plan's missing remainder only. **Touches:** nothing directly — it only submits jobs.

```mermaid
flowchart TB
    R[sc reconcile] --> LK["take the reconcile lock<br/>(a second invocation no-ops)"]
    LK --> PZ2{"line paused?"} -- yes --> HH["stop: human review first"]
    PZ2 -- no --> OBS["observe desired (plan) vs<br/>actual (run tree + queue)"]
    OBS --> D{per stage}
    D -- "gate passes" --> NXT[next stage]
    D -- "jobs in flight" --> WAIT["wait — never race live work"]
    D -- "trained, unjudged" --> SE["submit the missing eval<br/>(retry budget 3)"]
    D -- "segments incomplete" --> SC2["resume the consolidate:<br/>finished segments SKIPPED,<br/>only the remaining Mtok train —<br/>cost = the plan's unspent part,<br/>nothing is redone (budget 3)"]
    D -- "judged and still failing" --> PV["VERDICT failure -><br/>pause for review"]
```

The recovery verb: level-triggered convergence, safe to invoke at any
moment from any trigger. It replaced the resume/reset/continue decision
tree — decisions come from observed state, never from which event
fired. Infra gaps get a bounded retry budget; real verdicts get a
human.

## Judging

### `sc eval RUNDIR…`

Judges specific checkpoints: one wide (8-way sharded) eval job per run dir, each with its own janitor.

**Cost:** one node × ~10 min per checkpoint. **Touches:** RUNDIR/eval/.

```mermaid
flowchart LR
    RD["checkpoint dir(s)"] --> WE["wide eval per checkpoint:<br/>one node, sharded across<br/>its 8 GPUs, ~10 min"]
    WE -- afternotok --> EJ2[evaljanitor]
```

### `sc sweep [--dense]`

Finds every checkpoint that has a model but no verdict and judges them all — wide by default, packed 8-per-node with `--dense`.

**Cost:** one node × ~10 min per gap (wide); --dense packs 8 checkpoints per node — same GPU-h, longer wall. **Touches:** each gap's eval/.

```mermaid
flowchart TB
    SW[sc sweep] --> SCAN["scan every run root for<br/>checkpoints with a model<br/>but no verdict (eval gaps)"]
    SCAN -- none --> N0["no eval gaps"]
    SCAN --> MODE{--dense?}
    MODE -- no --> WIDE["one wide eval per gap —<br/>same GPU-h as dense,<br/>~8x less wall-clock,<br/>small jobs backfill well"]
    MODE -- yes --> DENSE["chunks of 8 per node,<br/>short generation cap —<br/>truncation at the cap IS<br/>the verdict 'broken'"]
```

### `sc bench TAG --model M --suite S`

Evaluates any model directory against any suite through the standard judge — for baselining foreign models or re-scoring ours on new suites.

**Cost:** one wide-eval node-run. **Touches:** runs/bench/TAG only.

```mermaid
flowchart LR
    BM["any model dir<br/>x any suite"] --> CK2["both must exist<br/>(checked login-side)"]
    CK2 --> BE["standard wide eval: same<br/>judge, same repair semantics;<br/>foreign models skip the<br/>retention guard"]
    BE --> ROW["a battery row"]
```

## Data

### `sc replay [COUNT]`

Regenerates the retention band at scale: the base model answers plain-C++ prompts, the compiler keeps verified pairs. The anchor data that stops fine-tuning from eroding ordinary competence.

**Cost:** one single-node generation job (scales with COUNT). **Touches:** the replay band artifact.

```mermaid
flowchart LR
    RP[sc replay] --> GJ["generation job: the BASE model<br/>answers ~COUNT training-shaped<br/>plain-C++ prompts"]
    GJ --> OV["compiler verifies every answer"]
    OV --> RB["retention band at scale —<br/>run before the first consolidation<br/>(a small band cannot anchor a<br/>large corpus)"]
```

### `sc harvest TAG [--nodes N --samples K --temp T]`

Expert-iteration burst: the model samples best-of-K answers to fresh prompts across N independent single-node jobs; oracle-verified winners become the next trainpack's expert band.

**Cost:** N single-node jobs for the sampling wall-clock; re-runs pay only unfinished shards. **Touches:** runs/harvest/TAG.

```mermaid
flowchart TB
    HV[sc harvest] --> CK3["model + prompts must exist;<br/>prompts are FRESH — never<br/>the eval suite"]
    CK3 --> FLEET["N independent single-node jobs<br/>(fleet-of-ones: queue-friendly,<br/>no collectives, no wide unknowns)"]
    FLEET --> BN["each: best-of-K sampling<br/>per prompt shard"]
    BN --> OR{"oracle:<br/>compile+run"}
    OR -- winner --> W["winners append live;<br/>.done marker per shard"]
    OR -- none pass --> DR[dropped]
    W --> RES["re-running the same command<br/>re-queues only unfinished shards"]
```

### `sc corpusfetch`

Prefetches every registry package to shared storage from the login node, because compute nodes have no internet.

**Cost:** login-node bandwidth only. **Touches:** the corpus source staging dir.

```mermaid
flowchart LR
    CF[sc corpusfetch] --> RG["read the staged generator's<br/>package registry"]
    RG --> CL["shallow-clone every entry<br/>login-side (compute nodes<br/>have no internet)"]
    CL --> ID["idempotent: existing<br/>checkouts reused"]
```

### `sc corpus`

Submits the corpus-factory job that harvests the prefetched packages into verified training records — after checking every precondition login-side.

**Cost:** one CPU-partition job (no GPUs). **Touches:** the corpus output file, append-only.

```mermaid
flowchart TB
    CO[sc corpus] --> PRE["ALL preconditions login-side:<br/>generator staged? config? vendored<br/>capture tool? sources fetched?<br/>container present? no factory<br/>already queued?"]
    PRE -- any missing --> BL["BLOCKED: each problem printed,<br/>zero core-hours spent"]
    PRE -- all present --> FJ["submit the corpus-factory job<br/>(CPU partition)"]
```

### `sc keep GEN/ROUND-sSEED…`

Re-derives a lost keeper checkpoint by retraining its recorded recipe. The result is a new sample of that recipe, judged by its own fresh eval.

**Cost:** one full round retrain + eval per keeper — the most expensive recovery verb; use only for checkpoints worth that price. **Touches:** the run dir + retention tier on success.

```mermaid
flowchart LR
    KP[sc keep] --> SPEC["look up the run's<br/>recorded chain spec<br/>(recipe: mix, drill, steps, seed)"]
    SPEC -- missing --> NO["cannot re-derive"]
    SPEC --> RT["retrain the exact recipe<br/>+ chained eval"]
    RT --> HON["HONESTY: runs are not<br/>bit-deterministic — this is a NEW<br/>SAMPLE; its fresh eval decides its<br/>worth, the old number does not carry"]
```

## Observing (read-only)

All observation verbs are free: login-side reads of the queue, the
run tree, and accounting — no jobs, no writes.

### `sc status`

The one-look campaign dashboard: rates, burn, failures, gaps, and what is running — ordered so the most important line lands next to the prompt.

```mermaid
flowchart TB
    ST2[sc status] --> TOP["scrolls away first: latest eval<br/>rates, recently finished jobs,<br/>24h node-h burn + top eaters"]
    TOP --> MID2["second-to-last: failures,<br/>eval gaps, excluded nodes,<br/>campaign stage, storage weather"]
    MID2 --> BOT["at the prompt (most visible):<br/>pause banner + RUNNING/queued<br/>with backfill start estimates"]
    NOTE["bottom-up terminal order:<br/>what you must see sits<br/>next to your cursor"] -.- BOT
```

### `sc status-all` · `sc health` · `sc jobstate JOBID`

Drill-downs: `status-all` adds a process-level look inside every running job, `health` probes running jobs for hangs, `jobstate` inspects one job.

```mermaid
flowchart LR
    SA[status-all] --> ST3[status] --> EACH["+ per running job:<br/>process-level look"]
    H[health] --> RJ2["every running job probed<br/>for hang signals"]
    JS[jobstate JOBID] --> ONE["one job: training metrics tail,<br/>eval slot states, GPU busy"]
```

### `sc report GEN [REF] [PREV]`

Prints the decision contract for a generation — the numbers a go/no-go is actually made on, with reference and previous-generation columns.

```mermaid
flowchart LR
    RE[sc report] --> RD2["read the generation's evals<br/>(+ base reference, + previous gen)"]
    RD2 --> DC["the DECISION CONTRACT per config:<br/>seeds, suite mean/min-max/sigma,<br/>guard-min, first-shot vs repaired"]
```

### `sc quota`

Everything spendable in one read-only view: billing units, storage on both tiers, node ceilings, idle capacity, and start estimates for burst shapes.

```mermaid
flowchart LR
    QU[sc quota] --> RO["read-only sweep: billing units,<br/>storage on both tiers, per-job node<br/>ceilings, idle nodes right now"]
    RO --> EST["start estimates for burst<br/>shapes (--test-only probes)"]
```

### `sc weatherprobe` · `sc weather`

Storage-weather instrumentation: `weatherprobe` is the scheduled background probe that writes the verdict submissions consult; `weather` mines our own job logs into an hour-of-day stall grid.

```mermaid
flowchart LR
    WP[weatherprobe] --> BP["bounded probe of the active<br/>storage tier (background job,<br/>scheduled ~5 min)"]
    BP --> VJ2["verdict file: submissions consult<br/>it instantly — patience belongs in<br/>the QUEUE, not on billed nodes"]
    WA[weather] --> TG["mine our own job logs'<br/>stall sentinels"]
    TG --> GRID["recurrence grid by hour-of-day —<br/>the shared-filesystem schedule<br/>emerges from normal operation;<br/>hand the grid to the operator"]
```

### `sc watch` · `sc bg VERB…`

`watch` is a finite collect-everything loop (drain, sweep, drain, report); `bg` detaches any verb so it survives logout.

```mermaid
flowchart LR
    W2["sc bg watch GEN"] --> DT["re-exec detached<br/>(survives logout)"]
    DT --> LOOP2["finite loop: drain queue -><br/>sweep gaps -> drain -> report"]
```

## Recovery

Janitors and resume are ~free (CPU-minutes and a file write); the only
expensive recovery is `keep`, priced above.

### `sc janitor` · `sc evaljanitor NAME`

The two automatic responders: `janitor` recovers node-fault failures (the only auto-retry class), `evaljanitor` retries a failed eval exactly once before pausing the line.

```mermaid
flowchart TB
    F{failed job} -- "node fault<br/>(the ONLY auto-retry)" --> JN2["janitor: resubmit the chain,<br/>add the node to the exclude list,<br/>REPORT it upstream — not<br/>just route around"]
    F -- "anything else" --> P2["pause the line: queued jobs<br/>no-op, submissions refuse —<br/>errors are information"]
    EVF{failed eval} --> EJ3["evaljanitor: verdict already<br/>on disk? done. else retry ONCE<br/>(shard-resume reuses finished<br/>work); second failure -> pause"]
```

### `sc resume`

Clears the pause after human review and lists what still needs judging.

```mermaid
flowchart LR
    RS2[sc resume] --> CL2["clear the pause<br/>(after human review)"]
    CL2 --> GAPS["list checkpoints still<br/>unjudged -> sc sweep"]
```

## Storage doctrine

All storage verbs are free of compute — filesystem moves, symlinks,
and deletes; `gc` is dry-run by default and never touches evidence.

### `sc clean` · `sc archive` · `sc gc [--delete]`

Storage-doctrine enforcement: `clean` retires judged checkpoints and folds strays home, `archive` pushes superseded artifacts to the retention tier, `gc` reclaims weight space while always keeping the scientific record.

```mermaid
flowchart TB
    CLN[sc clean] --> DOC["doctrine: working tier = records +<br/>in-flight checkpoints; retention<br/>tier = evaluated ones"]
    DOC --> RET["retire evaluated checkpoints<br/>(symlink left behind)"]
    DOC --> SAL["delete weight-bearing salvage"]
    DOC --> FOLD["fold stray scratch trees home"]
    AR[sc archive] --> SUP["superseded artifacts only:<br/>sealed segments' predecessors,<br/>judged round checkpoints"]
    SUP --> MV["push to retention, symlink back,<br/>idempotent — nothing hot touched"]
    GC[sc gc] --> SCAN2["scan every run's weights;<br/>KEEP: release, current best,<br/>anything touched < 24h"]
    SCAN2 --> DRY["dry-run report: reclaimable GiB"]
    DRY -- "--delete" --> DEL["remove weights of superseded runs;<br/>the scientific record (evals, metrics,<br/>manifests, validations) is ALWAYS kept"]
```

---

Inside the jobs: the open-source harness tools in this repo —
flowcharts in [tools.md](tools.md). Dataset-side commands (`cpp26ds`:
drillgen, synthgen, harvest-packages, trainpack, …):
[storax-dataset-cpp26 docs/cli.md](https://github.com/storax-io/storax-dataset-cpp26/blob/main/docs/cli.md).
