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
| two-model study: training set | `--layout-seed 11 --layout-per-episode` | `iid11` | 400 (100 per task) |
| two-model study: test set | `--layout-seed 12 --layout-per-episode` | `iid12` | 400 (100 per task) |

In the first three sets, a task's episodes share a few turns of the object: SimplerEnv's layouts have 1 to 3
turns per task, and a `--layout-seed` set has 12 placements per task, each with the same 4 turns. In the `iidN`
sets, each episode has a random placement and turn of its own (`IID_EPISODES` sets the episodes per task).

Planner-only evals are named `{prefix}_{layouts}_{level}`, with the depth-noise level `none` (clean),
`realistic` or `harsh`. The current ones are `pseed_*` (planner seeded per episode). The VLA-only evals:
`fast3_train_g0/g1` (test set), `heldout_vla_g0/g1` (training set), `seed2_vla` (held-out test set), `vla_iid11`
and `vla_iid12`, all with the fast settings, and the reruns `rep1_*`, `rep2_*` (VLM seed offsets 1 and 2).
`exact_vla_1` to `exact_vla_5` link to the 5 VLA-only evals of the test set with the paper's settings
(`run_eval.sh`). `runs.py` has these names.

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
# the same eval with SAM 3.1's detector for the visual prompts (needs the sam31 env, build_env_sam31.sh)
SAM_VERSION=sam3.1 SEED_OFFSET=1 bash $E/vla_eval.sh 0 10460 sam31_rep1_paper paper 4
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

The two-model study's evals and ASMs (see "Two-model ASM study" below):

```bash
bash $E/planner_evals.sh 0 10320 3 pseed_iid11_none:none:iid11 pseed_iid11_realistic:realistic:iid11 \
    pseed_iid11_harsh:harsh:iid11 pseed_iid12_none:none:iid12 pseed_iid12_realistic:realistic:iid12 \
    pseed_iid12_harsh:harsh:iid12
HIDDEN=1 PERTURB=8 bash $E/vla_eval.sh 0 10400 vla_iid11 iid11 4
HIDDEN=1 PERTURB=8 bash $E/vla_eval.sh 0 10400 vla_iid12 iid12 4
HIDDEN=1 PERTURB=8 SIM_ARGS="--sam3-batch-prompts --no-action-plots --max-episode-steps 1" \
    bash $E/vla_eval.sh 0 10420 perturb_paper paper 4
mkdir -p $RUNS_DIR/two_model
for level in none realistic harsh; do
  python -m examples.SimplerEnv.eval_files.planner.selector --rule two-model \
      --planner-runs $RUNS_DIR/pseed_iid11_$level --vla-runs $RUNS_DIR/vla_iid11 \
      --save $RUNS_DIR/two_model/asm_$level.json > $RUNS_DIR/two_model/train_$level.txt
done
DEPTHS=harsh CHUNK_SIZE=25 bash $E/selector_eval.sh 0 10510 closed_loop_iid12 iid12 'two_model/asm_{}.json' 6
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

`compare_runs RUN_A:RUN_B...` compares two planner-only evals episode by episode. The two-model study's scorers:
`score_two_model` (all its tables and tests) and `check_closed_loop closed_loop_iid12 --planner 'pseed_iid12_{}'
--models 'two_model/asm_{}.json' --vla vla_iid12` (its closed-loop eval).

The `dfix_*` and `seeded_*` runs and `heldout_planner_none` (the evals before the planner seed and before the
depth fix) were archived on 2026-10-15; restore them into `$RUNS_DIR` for the result files that read them.
`dfix_scores_2026-10-14.txt` was made by an older version of `score_change` with these "before" evals:
`seeded_paper_{level}`, `heldout_planner_none`, `seeded_heldout_{realistic,harsh}`. Link them as
`pre_dfix_paper_{level}` and `pre_dfix_heldout_{level}` to get the same numbers (a few headings and run names
in the first table differ).

## Two-model ASM study

The plan below was fixed on 2026-10-15, before the study's evals ran, and `score_two_model` follows it. The
results do not change it. A result that is not in the plan is reported as exploratory.

**Question.** The two-model ASM gives an episode to the planner when P(planner succeeds) >= P(VLA succeeds), else
to the VLA. Does it get a higher success rate than either method alone, and than rules that do not estimate the
VLA's chance in each episode?

**Why new layout sets.** In the old sets, a task's episodes share 1 to 4 turns of the object (see the table above).
The VLA's success depends mostly on the turn (test set: Spoon 100% at 0 deg and 33% at 90 deg; Carrot 92% at 180 deg
and 3% at -90 deg), so a set has only 1 to 4 independent VLA conditions per task, and a task's VLA success rate
changes a lot between sets (Eggplant: 63% on the training set, 88% on the test set). A VLA success model cannot
learn the turn from so few values. In `iid11` (training set) and `iid12` (test set), each episode has a layout of
its own: one placement and one turn from the same draw as before (`draw_layouts` in `planner_evaluator.py`), with a
random generator seeded by the layout seed and the episode number. Each set has 100 episodes per task.

**Evals.** Planner-only evals `pseed_iid11_{level}` and `pseed_iid12_{level}` (planner seed 0, the 3 depth
levels); VLA-only evals `vla_iid11` and `vla_iid12` (fast settings, `HIDDEN=1 PERTURB=8`); `perturb_paper`
(1-step episodes of the test set, for the action spread there); the closed-loop eval `closed_loop_iid12` (harsh
depth) with the saved ASMs `two_model/asm_{level}.json`. The commands are above.

**Models.** One ASM per depth level, trained on `iid11` only. Each success model is a logistic regression (L2
penalty 1, standardized inputs) on its features and one column per task (the tasks are the same in training and
test; only the layouts are new). A first-look feature that is missing (the object was not found) is 0, as in the
earlier ASMs.

- P(planner succeeds): `plan_ok` and `plan_path_ratio`, the v3 features (with harsh depth, the only planner
  features with signal on the old sets).
- P(VLA succeeds): `grasp_turn_rad` (how far the hand must turn for a top-down grasp across the object) and
  `grasp_reach_m` (how far the object is from the hand). Together they give the layout: how the object is turned
  and where it is. Both come from the planner's first look (camera only), before the robot moves. The turn was
  the strongest VLA signal on the old sets (within-task AUC 0.84 on the test set).

**Decision rule.** The planner when P(planner) >= P(VLA), else the VLA. When both probabilities are calibrated, no
other choice gives a higher expected success.

**Planned tests.** On the 400 episodes of `iid12` with harsh depth (with harsh depth the planner fails most, so
the choice matters most): the two-model ASM against, in this order,

1. the better single method on the training set (planner-only or VLA-only);
2. rule A: the planner when P(planner) >= 0.5, with the same P(planner) (the gain from estimating P(VLA));
3. rule C: the planner when P(planner) >= the task's VLA success rate on the training set (the gain from the
   episode's VLA features).

Each test is the exact two-sided paired sign-flip test (with one VLA run per episode, the exact McNemar test) at
0.05. A test counts only when every test before it was significant, so the chance of any false claim stays at
0.05 or less. Intervals: 95% percentile bootstrap; each draw takes each task's episodes again, with replacement
(10,000 draws).

**Secondary (reported, no claims).** The clean and realistic depth levels; the test set (`paper`: it was used to
choose the VLA features, so it is not a clean test) and the held-out test set (`seed2`: its 4 turns per task make
it a weak test of the turn); P(VLA) with the log action spread added (the standard deviation of the first action
chunk over 8 changed copies of the input, `vla_perturbation.py`); the linear probe's P(VLA) (last layer, PCA to 10
numbers, as planned before); the previous ASM v3 and the heuristic baseline; the success models' AUC, within-task
AUC, log-loss, Brier score and calibration (ECE, 10 equal bins) against the task's rate; the closed-loop eval
against the offline result (`check_closed_loop`).
