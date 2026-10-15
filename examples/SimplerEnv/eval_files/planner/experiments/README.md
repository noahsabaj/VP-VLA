# Adaptive selector experiments

The scripts that made the adaptive selector (ASM) results: eval runners (`*.sh`) and scorers (`*.py`).
Each eval writes one run folder in `$RUNS_DIR`; each scorer reads run folders and prints one result file.

## Setup

- `RUNS_DIR`: the folder with the run folders. Run names in the commands below are folders in it; absolute
  paths also work.
- The eval environments, as for `auto_eval_scripts/run_eval.sh`: `CONDA_ENVS_ROOT` (or `STARVLA_PYTHON`,
  `SAM3_PYTHON`, `SIMPLER_PYTHON`) and `SIMPLERENV_PATH` (see `examples/eval_env.sh`).
- `CKPT`: the VP-VLA SimplerEnv checkpoint (default
  `playground/Checkpoints/VP-VLA-OXE/checkpoints/steps_70000_pytorch_model.pt`).

The runners start from any folder. Run the scorers from the repository root, with numpy:

```bash
python -m examples.SimplerEnv.eval_files.planner.experiments.SCORER ...
```

## Layout sets and run names

| Set | Layouts | `LAYOUTS` | Episodes |
|---|---|---|---|
| training set | `--layout-seed 1` | `heldout` | 192 (48 per task) |
| test set | SimplerEnv's fixed layouts (the paper's eval) | `paper` | 96 (24 per task) |
| held-out test set | `--layout-seed 2` | `seed2` | 192 (48 per task) |

Planner-only evals are named `{prefix}_{layouts}_{level}`, with the depth-noise level `none` (clean),
`realistic` or `harsh`. The current ones are `pseed_*` (planner seeded per episode). The VLA-only evals:
`fast3_train_g0/g1` (test set), `heldout_vla_g0/g1` (training set), `seed2_vla` (held-out test set), all with
the fast settings, and the reruns `rep1_*`, `rep2_*` (VLM seed offsets 1 and 2). `exact_vla_1` to `exact_vla_5`
link to the 5 VLA-only evals of the test set with the paper's settings (`run_eval.sh`).

## Evals (GPU)

| Runner | What it runs |
|---|---|
| `planner_eval.sh GPU PORT NAME LEVEL LAYOUTS` | one planner-only eval (one SAM3 server; the 4 tasks at the same time) |
| `planner_evals.sh GPU PORT_BASE AT_ONCE NAME:LEVEL:LAYOUTS...` | several planner-only evals at the same time |
| `vla_eval.sh GPU PORT_BASE NAME LAYOUTS K [TASK:START:END...]` | one VLA-only eval (one server set, K sim clients) |
| `selector_eval.sh GPU PORT_BASE NAME LAYOUTS MODELS K [LEVEL:TASK:START:END...]` | the closed-loop ASM eval |
| `sim_task.sh GPU TASK START END OUT [ARGS...]` | one sim run (smoke tests; the servers must already run) |

Each script's header lists its options. The VLA runners use the fast settings by default (`FAST=1`, see
`fast_defaults` in `common.sh`; `FAST=0`: the paper's settings) and an MPS daemon of their own (`MPS=1`). A
run's `status.log` has a START line with its exact settings and commit, one line per chunk or task, and a DONE
line. Examples:

```bash
E=examples/SimplerEnv/eval_files/planner/experiments
# planner-only evals, 3 at a time (PLANNER_SEED=0 is the default)
bash $E/planner_evals.sh 0 10320 3 pseed_heldout_none:none:heldout pseed_heldout_realistic:realistic:heldout \
    pseed_heldout_harsh:harsh:heldout pseed_paper_none:none:paper pseed_paper_realistic:realistic:paper \
    pseed_paper_harsh:harsh:paper pseed_seed2_none:none:seed2 pseed_seed2_realistic:realistic:seed2 \
    pseed_seed2_harsh:harsh:seed2
# VLA-only evals (chunks of 8 episodes, K=4 sim clients); the reruns with SEED_OFFSET=1 and 2
HIDDEN=1 bash $E/vla_eval.sh 0 10400 seed2_vla seed2 4
SEED_OFFSET=1 HIDDEN=1 bash $E/vla_eval.sh 0 10460 rep1_paper paper 4
# first-step hidden states for the linear probe (1-step episodes)
HIDDEN=1 SIM_ARGS="--sam3-batch-prompts --no-action-plots --max-episode-steps 1" \
    bash $E/vla_eval.sh 0 10420 probe_paper paper 4
# the ASM models: trained on the training set, tested on the test set
for level in none realistic harsh; do
  python -m examples.SimplerEnv.eval_files.planner.selector --features v3 \
      --planner-runs $RUNS_DIR/pseed_heldout_$level --vla-runs $RUNS_DIR/heldout_vla_g0 $RUNS_DIR/heldout_vla_g1 \
      --test-planner-runs $RUNS_DIR/pseed_paper_$level \
      --test-vla-runs $RUNS_DIR/fast3_train_g0 $RUNS_DIR/fast3_train_g1 \
      --save $RUNS_DIR/final_pseed/sel_v3_$level.json > $RUNS_DIR/final_pseed/train_$level.txt
done
# the closed-loop eval of the test set at the 3 levels
bash $E/selector_eval.sh 0 10510 live_pseed paper final_pseed 6
```

The two-GPU VLA evals give each GPU half of the episodes as explicit chunks, e.g. `heldout_vla_g0` has
`TASK:0:8 TASK:8:16 TASK:16:24` for each task and `heldout_vla_g1` the episodes 24 to 47.

## Result files (CPU)

| Result file in `$RUNS_DIR` | Scorer |
|---|---|
| `pseed_scores_2026-10-14.txt` | `score_change --before dfix --after pseed` |
| `pseed_seed2_scores_2026-10-14.txt` | `score_heldout_test --runs pseed` |
| `pseed_seed2_mean_scores_2026-10-14.txt` | `score_heldout_test --runs pseed --vla seed2_vla rep1_seed2 rep2_seed2` |
| `reps_scores_2026-10-14.txt` | `score_vla_seeds --runs pseed` |
| `feature_evidence_pseed_2026-10-14.txt` | `feature_evidence --runs pseed --paper-vla fast3_train_g0 fast3_train_g1 rep1_paper rep2_paper --heldout-vla heldout_vla_g0 heldout_vla_g1 rep1_heldout rep2_heldout` |
| `feature_evidence_2026-10-12.txt` | `feature_evidence --runs seeded` |
| `probe_mean_2026-10-14.txt` | `linear_probe --planner 'pseed_{}_harsh' --set paper=probe_paper:fast3_train_g0,fast3_train_g1,rep1_paper,rep2_paper --set heldout=probe_heldout:heldout_vla_g0,heldout_vla_g1,rep1_heldout,rep2_heldout --set seed2=seed2_vla/hidden:seed2_vla,rep1_seed2,rep2_seed2` |
| `live_pseed/live_check_2026-10-14.txt` | `check_closed_loop live_pseed --planner 'pseed_paper_{}' --models final_pseed` |
| `dfix_scores_2026-10-14.txt` | `score_change --before pre_dfix --after dfix` (see below) |

`compare_runs RUN_A:RUN_B...` compares two planner-only evals episode by episode.

The `dfix_*` and `seeded_*` runs and `heldout_planner_none` (the evals before the planner seed and before the
depth fix) were archived on 2026-10-15; restore them into `$RUNS_DIR` for the result files that read them.
`dfix_scores_2026-10-14.txt` was made by an older version of `score_change` with these "before" evals:
`seeded_paper_{level}`, `heldout_planner_none`, `seeded_heldout_{realistic,harsh}`. Link them as
`pre_dfix_paper_{level}` and `pre_dfix_heldout_{level}` to get the same numbers (a few headings and run names
in the first table differ).
