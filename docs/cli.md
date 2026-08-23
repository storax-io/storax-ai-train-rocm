# `sc` — the campaign CLI (usage reference)

`sc` is how the storax training campaign is operated on the
supercomputer: one command turns an intent — "run this generation",
"judge every checkpoint", "train 1B tokens as a segment chain" — into
a Slurm job chain. **Closed source, open usage**: the implementation is
private; this page documents the command surface and its semantics.

Design rules that hold for every verb: everything submitted is a plain
Slurm job — chains via `afterok`, failure janitors via `afternotok`,
**no daemons, no cron**. Errors pause the line; hardware faults
self-heal; nothing burns to a time cap. Every submit verb is
**idempotent** — re-running the same command resumes or converges
instead of duplicating work. The jobs themselves execute the
open-source harness tools in this repo ([tools.md](tools.md)).

```mermaid
flowchart TB
    subgraph gate["every submission passes"]
        PZ{"line paused?"} -- yes --> DIE["refuse: review,<br/>then sc resume"]
        PZ -- no --> WX{"fresh weather<br/>verdict bad?"}
        WX -- "storage dark<br/>< 15 min ago" --> DIE2["WEATHER HOLD<br/>(billed nodes never<br/>probe out a storm)"]
        WX -- clear/stale --> OKG[submit]
    end
    subgraph submitters["submit verbs"]
        S1["gen / rounds / eval / sweep"]
        S2["consolidate / campaign / reconcile"]
        S3["harvest / bench / replay / keep"]
        S4["corpus / corpusfetch"]
    end
    subgraph observers["observe verbs (read-only)"]
        O1["status / status-all / health /<br/>jobstate / report / quota /<br/>weather / watch"]
    end
    subgraph maint["recover + storage"]
        M1["janitor / evaljanitor / resume"]
        M2["clean / archive / gc"]
    end
    submitters --> gate --> Q[("Slurm queue<br/>afterok chains")]
```

## Launching training

### `sc preflight` · `sc gen MANIFEST` · `sc rounds GEN ROUND MIX DRILL STEPS SEED…`

```mermaid
flowchart LR
    MF["manifest rows:<br/>gen round mix drill seed steps"] --> PF["preflight (no cost):<br/>verify code + data vs<br/>the sync manifest"]
    PF --> RJ["round job per row"]
    RJ -- afternotok --> JN["janitor job<br/>(node faults only:<br/>retry + exclude list)"]
    RJ -- afterok --> EV["WIDE eval — own node,<br/>8-way sharded, ~10 min<br/>(EVAL SLO: no verdict<br/>layer runs an hour)"]
    EV -- afternotok --> EJ["evaljanitor: retry ONCE<br/>(shard-resume reuses work);<br/>second failure -> pause"]
```

`gen` reads a manifest (one row per round: generation, round name, mix,
drill share, seed, steps) and queues the whole generation — rounds,
per-round wide evals, janitors — then it is safe to log out. `rounds`
is the single-recipe form for a list of seeds.

### `sc consolidate NAME MIX DRILL SEED [MTOK]` — segment-chain training

```mermaid
flowchart TB
    LAD["width ladder (e.g. 8->16->32->64 nodes,<br/>custom via --ladder n:mtok,…)<br/>one LR schedule across segments"] --> SEGS["afterok segment chain<br/>(+1 spare node per segment: a sick<br/>node idles instead of killing the run)"]
    SEGS --> IDEM{"segment state<br/>already on disk?"}
    IDEM -- yes --> SKIP["skipped — re-running the<br/>same command IS the<br/>hang/failure recovery"]
    IDEM -- "mid-checkpoint" --> MID["resume from the segment's<br/>mid-checkpoint"]
    SEGS --> CEVAL["per-segment evals run<br/>CONCURRENTLY on their own nodes —<br/>the chain never waits on a verdict"]
    CEVAL --> CONV["converge director (tiny CPU job):<br/>seg-over-seg gain < --improve-min<br/>-> cancel the still-PENDING tail.<br/>The plateau is the cutoff, not the<br/>token plan. A missing eval<br/>cancels NOTHING."]
    CTRL["--controlled: submit only the next<br/>segment; each director computes<br/>family weights from its eval and<br/>extends the chain itself"] -.-> SEGS
    FORK["--from DIR: fork any validated<br/>checkpoint into a parallel branch"] -.-> SEGS
```

### `sc campaign PLAN.json` · `sc reconcile` — unattended multi-stage arcs

```mermaid
flowchart TB
    PL["plan: JSON stage list<br/>(gen | consolidate | replay | sweep),<br/>each with an artifact gate"] --> ST["submit stage i"]
    ST --> DIR["director (tiny CPU job,<br/>afterany on the stage's<br/>terminal jobs)"]
    DIR --> GATE{"gate on artifacts:<br/>any (rate, guard, medtok)<br/>candidate passes?"}
    GATE -- pass --> NEXT["submit stage i+1"]
    GATE -- fail --> PAU["pause with the reason"]
    REC["sc reconcile — THE recovery verb.<br/>Level-triggered: observe desired (plan)<br/>vs actual (run tree + queue), submit<br/>exactly what's missing, idempotently.<br/>INFRA gap -> resubmit (retry budget);<br/>VERDICT failure -> pause for review.<br/>Safe at any moment, from any trigger."] -.-> GATE
```

Campaign state is keyed to the plan (a new plan starts at stage 0); a
stale director continuation is ignored. `reconcile` replaced the whole
resume/reset/continue decision tree: decisions come from observed
state, never from which event fired.

## Judging

### `sc eval RUNDIR…` · `sc sweep [--dense]` · `sc bench TAG --model M --suite S`

| verb | semantics |
|---|---|
| `eval` | wide eval per checkpoint: one node, sharded across its 8 GPUs, ~10 min — same GPU-h as dense, ~8× less wall-clock, small jobs backfill well |
| `sweep` | find every checkpoint with a model but no verdict and submit wide evals for all of them; `--dense` packs 8 per node with a short generation cap — a truncation at the cap *is* the verdict "broken" |
| `bench` | any model × any suite through the same judge with the same repair semantics; foreign models skip the retention guard |

## Data verbs

| verb | semantics |
|---|---|
| `sc replay [COUNT]` | regenerate the retention band at scale: base-model answers to training-shaped plain-C++ prompts, compiler-verified — run before the first consolidation |
| `sc harvest TAG` | expert-iteration burst: N independent single-node jobs, best-of-N sampling on FRESH prompts (never the eval suite), the oracle keeps verified winners; fleet-of-ones queues well; re-run to resume unfinished shards |
| `sc corpusfetch` | login-side prefetch (compute nodes have no internet): shallow-clone every registry package for the corpus factory |
| `sc corpus` | submit the corpus-factory job — **all** preconditions checked login-side before a single core-hour is queued |

## Observing

| verb | one line |
|---|---|
| `sc status` | bottom-up terminal order: latest rates and 24h burn scroll away, failures sit second-to-last, RUNNING lands at the prompt; plus eval gaps, excluded nodes, campaign stage, storage weather |
| `sc status-all` | status + a process-level look inside every running job |
| `sc health` | probe running jobs for hangs |
| `sc jobstate JOBID` | peek inside one job: metrics tail, eval slot states, GPU busy |
| `sc report GEN [REF] [PREV]` | the generation decision contract: seeds, suite mean/min–max/σ, guard-min, first-shot vs repaired |
| `sc quota` | billing units, storage on both tiers, node ceilings, idle counts, start estimates for burst shapes — read-only |
| `sc weatherprobe` | background storage-weather probe (scheduled ~5 min); submissions consult the stored verdict instead of discovering the weather on a paid allocation |
| `sc weather` | stall-recurrence grid by hour-of-day, mined from our own job telemetry — the shared-filesystem schedule emerges from normal operation |
| `sc watch` | finite collection loop: drain → sweep gaps → drain → report; run detached via `sc bg watch …` |
| `sc bg VERB…` | re-exec any verb detached (survives logout) |

## Recovery and storage

```mermaid
flowchart LR
    F{failure} -- "node fault<br/>(the only auto-retry)" --> J["janitor: resubmit +<br/>exclude list; faulty nodes<br/>REPORTED upstream,<br/>not just excluded"]
    F -- "anything else" --> P["line pause: queued jobs no-op,<br/>submissions refuse —<br/>errors are information"]
    P --> HR["human review"] --> RS["sc resume: clear the pause,<br/>list unjudged checkpoints"]
```

| verb | semantics |
|---|---|
| `sc clean` | enforce the storage doctrine: working tier holds records + in-flight checkpoints, retention tier holds evaluated ones — retire (symlink back), delete salvage debris, fold strays home |
| `sc archive` | push superseded artifacts to the retention tier (sealed segments' predecessors, judged round checkpoints); symlinks left; idempotent |
| `sc gc [--delete]` | weights of superseded runs are disposable; the scientific record (evals, metrics, manifests, validations) is always kept; dry-run by default |
| `sc keep GEN/ROUND-sSEED…` | re-derive lost keeper checkpoints from their chain specs — honestly: runs are not bit-deterministic, a re-derived keeper is a NEW SAMPLE and its fresh eval decides its worth |

---

Inside the jobs: the open-source harness tools in this repo —
flowcharts in [tools.md](tools.md). Dataset-side commands (`cpp26ds`:
drillgen, synthgen, harvest-packages, trainpack, …):
[storax-dataset-cpp26 docs/cli.md](https://github.com/storax-io/storax-dataset-cpp26/blob/main/docs/cli.md).
