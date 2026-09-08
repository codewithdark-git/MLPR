# MLPR autoresearch program

This is the MLPR adaptation of karpathy/autoresearch: the LLM agent runs its
own research loop on the Knowing–Using Gap project. One GPU (Modal H100),
fixed training budget, one score, keep-or-discard ledger, autonomous loop.

## Setup (done once per run tag)

1. **Run tag**: `sep6` (date-based). Work happens on git branch
   `autoresearch/sep6` of the MLPR repo, branched from `main`.
2. **In-scope files** (read for full context before experimenting):
   - `idea.md` — the research proposal: math, dataset spec, metrics. The
     ground truth the implementation must match.
   - `main.py` — training entrypoint (AR knobs: `ar_mode`, `ar_time_budget_s`,
     `eval_*_n`).
   - `src/trainer/mlpr_trainer.py`, `src/callbacks/*.py`, `src/models/*.py`,
     `src/data/*.py` — the editable machinery.
   - `scripts/generate_synthetic_dataset_v2.py` — data generator (v2 quality
     upgrade: valid-answer sets, |A|=2000, reverse probing).
3. **Fixed harness (read-only — the `prepare.py` equivalent)**:
   - `src/evaluation/gen_eval.py` metric definitions (em / em_any / likelihood).
   - `dataset/`, `dataset_v2/` ground-truth JSONL + vocab.
   - The eval protocol inside `main.py` (baseline/final evaluation calls).
   The agent may not redefine a metric to make its number look better.
4. **Ledger**: `autoresearch/results.tsv` — one row per experiment.

## What you CAN change

Everything in `src/` (trainer, callbacks, models, data), `configs/`,
hyperparameters passed through `ar_experiments.py --overrides JSON`, the
dataset generator (with a new version dir + seed so ground truth stays
comparable), and `main.py` plumbing (without touching the eval protocol).

## What you CANNOT change

- The scoring functions in `src/evaluation/gen_eval.py`.
- The dataset ground truth (`label_text`, `valid_answers`, vocab ids) once an
  experiment with that dataset has been logged.
- The fixed-budget rule: `ar_time_budget_s` is identical when comparing two
  experiments unless the change itself is the budget.

## The score (this project's `val_bpb`)

Primary metric (lower-good `val_bpb` analogue, here higher-is-better):

    score = a_gen_em_any        # validity-aware multi-hop generalization

from the `---` summary block `ar_experiments.py` prints. Validity-aware means
an answer counts when it matches ANY entry of the record's `valid_answers`
set (v2 dataset + v3 scoring; multi-answer queries are no longer scored
against one arbitrary label).

**Short-budget tie-break (lexicographic).** At 360-600 s budgets multi-hop
EM is honestly ~0 (multi-hop needs BOTH atomic hops stored first), so when
`a_gen_em_any` ties at 0 across a comparison batch, rank by:
1. `a_mem_lf_any` — atomic "knowing" recall (the prerequisite),
2. `probe_mem_last` — mid-layer decodability (the causal intermediary the
   probe gradient shapes; fastest-moving honest signal),
3. lower `train_loss` (optimization health).
The primary objective stays a_gen; tie-breakers only order experiments that
have not yet exited the zero regime. Gate-threshold arms evaluated at short
budgets inform the EARLY-injection regime only; the production tau_0 is
decided in the 50-epoch v5 runs.

Guardrails (a change that violates either is a discard regardless of score):
- `a_mem_lf_any` must not drop more than 0.02 below the current champion's.
- `val_loss_last` must not exceed the champion's by more than 5%.
- No metric may be improved by disabling evaluation or shrinking eval sets
  (eval knobs must stay fixed within a comparison batch).

## Launching one experiment

    # Modal's --detach MUST come BEFORE the script path (it keeps the app
    # alive after the local CLI exits; after the path it would bind to the
    # entrypoint's own flag and the spawned queue gets canceled).
    modal run --detach ar_experiments.py --exp <name> \
        --overrides '{"tau_0": 0.9, "delta_0": 0.05}' \
        --dataset dataset_v2 --budget 360

    # Sequential queue inside one detached call (warm container):
    modal run --detach ar_experiments.py --queue exp_a,exp_b,exp_c \
        --override-list '{}||{"tau_0":0.25}||{}' --dataset dataset_v2

- The summary block is greppable: `modal app logs <id> | grep -A 14 "^---"`.
- Budget: **360 s** training wall clock (epoch-end granularity) + fast evals.
  A full experiment lands in ~8–12 min H100 time including load + eval.
- AR epoch ceiling: 12 epochs, warmup_ratio 0.05 (warmup must finish well
  inside the budget — an inflated epoch ceiling pushes warmup past the wall
  and the run stops before real LR ever applies).

## The experiment loop

LOOP FOREVER:

1. `git status` — confirm branch/commit state.
2. Pick ONE change (single-variable experiments only): config override, code
   tweak, or new data version.
3. `git commit` (small, descriptive).
4. Launch the experiment (command above); for queues use `--detach` and
   launch several sequentially chained inside one call.
5. Grep the summary block from the app logs; if the run crashed read the
   tail, fix, relaunch (a crash after 3 fix attempts => log `crash`, revert).
6. Append a row to `autoresearch/results.tsv`:
   `commit<TAB>score<TAB>mem_gb<TAB>status<TAB>description`
   (0.000000 score + status `crash` for failures).
7. If score improved over the champion AND guards hold: keep (branch advances).
   Else: `git reset --hard <prev>` and move on.

## Idea backlog (seeded by the audit of the v4 50-epoch runs)

1. baseline replication (v1 data, v4 config) — control row.
2. dataset_v2 (valid-answer scoring + 2000 entities + reverse probing).
3. proposal-exact gate: tau_0=0.9, delta_0=0.05 (idea.md defaults) vs the
   v4 calibration (0.8 / 0.1).
4. gate_metric=likelihood only (pure "knowing" signal) vs max.
5. probe LR schedule: constant 1e-3 vs cosine-decayed.
6. LoRA rank 16 -> 32 (capacity for compositional routing).
7. two-phase lambda: fast ramp (delta_0=0.05) after likelihood saturation.
8. dropout 0.05 -> 0.0 on LoRA (memorization regime).
9. combine the winners into the v5 config; verify vs champion with 2 seeds.

## NEVER STOP

Once the loop begins, do not pause to ask the human. If out of ideas,
re-read `idea.md`, combine near-misses, or re-seed the champion. The loop
runs until interrupted. GPU is finite: prefer 360 s budgets, chain detached
runs, and never leave an H100 idle while a decision is pending.
