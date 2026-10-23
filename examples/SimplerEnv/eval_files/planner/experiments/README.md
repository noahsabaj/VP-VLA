# Adaptive selector experiments

The scripts that made the adaptive selector (ASM) results: eval runners (`*.sh`) and scorers (`*.py`).
Each eval writes one run folder in `$RUNS_DIR`; each scorer reads run folders and prints one result file.

## Setup

- `RUNS_DIR`: the folder with the run folders. Run names in the commands below are folders in it; absolute
  paths also work.
- The eval environments, as for `auto_eval_scripts/run_eval.sh`: `CONDA_ENVS_ROOT` (or `STARVLA_PYTHON`,
  `SAM3_PYTHON`, `SIMPLER_PYTHON`, and `SAM31_PYTHON` for SAM 3.1, built by
  `examples/Robocasa_tabletop/visual_prompt_utility/build_env_sam31.sh`) and `SIMPLERENV_PATH` (see
  `examples/eval_env.sh`).
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
| plan-check v2 study: test set | `--layout-seed 13 --layout-per-episode` | `iid13` | 96 (24 per task); harder scenes 240 (48 per task) |
| learned-features data: second training set | `--layout-seed 14 --layout-per-episode` | `iid14` | 400 (100 per task) |

In the first three sets, a task's episodes share a few turns of the object: SimplerEnv's layouts have 1 to 3
turns per task, and a `--layout-seed` set has 12 placements per task, each with the same 4 turns. In the `iidN`
sets, each episode has a random placement and turn of its own (`IID_EPISODES` sets the episodes per task). The
harder scenes take `paper` (their fixed layouts) or `iidN` (`harder_scenes.random_layout`).

Planner-only evals are named `{prefix}_{layouts}_{level}`, with the depth-noise level `none` (clean),
`realistic` or `harsh`. The current ones are `pseed_*` (planner seeded per episode). The VLA-only evals:
`fast3_train_g0/g1` (test set), `heldout_vla_g0/g1` (training set), `seed2_vla` (held-out test set), `vla_iid11`
and `vla_iid12`, all with the fast settings, and the reruns `rep1_*`, `rep2_*` (VLM seed offsets 1 and 2).
`exact_vla_1` to `exact_vla_5` link to the 5 VLA-only evals of the test set with the paper's settings
(`run_eval.sh`). `runs.py` has these names. The camera-noise study's evals are `cam_{layouts}_{level}`
(planner-only) and `vla_{layouts}_{level}` (VLA-only), where the level is the camera's (see "Camera-noise study").
The reworded-instruction study's are `words_{set}_paper` and `vla_words_{set}_paper` (see "Reworded-instruction
study"). The harder-scene study's are `harder_paper` and `vla_harder_paper` (see "Harder-scene study"). The VLM
plan-check study adds `check_paper_none` (see "VLM plan-check study"). The VLM plan-check v2 study's are
`check2_{own,words_SET,harder}_{paper,iid13}` (planner-only, both checks) and `vla_{own,words_SET,harder}_iid13`
(VLA-only; see "VLM plan-check v2 study"). The learned-features data are `pseed31_{iid11,iid14}_harsh` and
`vla31_{iid11,iid14}` (SAM 3.1; see "Learned first-frame features").

## Evals (GPU)

| Runner | What it runs |
|---|---|
| `planner_eval.sh GPU PORT NAME LEVEL LAYOUTS` | one planner-only eval (one SAM server; the 4 tasks at the same time) |
| `planner_evals.sh GPU PORT_BASE AT_ONCE NAME:LEVEL:LAYOUTS...` | several planner-only evals at the same time |
| `vla_eval.sh GPU PORT_BASE NAME LAYOUTS K [TASK:START:END...]` | one VLA-only eval (one server set, K sim clients) |
| `selector_eval.sh GPU PORT_BASE NAME LAYOUTS MODELS K [LEVEL:TASK:START:END...]` | the closed-loop ASM eval |
| `sim_task.sh GPU TASK START END OUT [ARGS...]` | one sim run (smoke tests; the servers must already run) |

Each script's header lists its options. The VLA runners use the fast settings by default (`FAST=1`, see
`fast_defaults` in `common.sh`; `FAST=0`: the paper's settings) and an MPS daemon of their own (`MPS=1`). A
run's `status.log` has a START line with its exact settings and commit, one line per chunk or task, and a DONE
line. `RGB_NOISE=realistic|harsh` puts a real color camera's errors (`camera_noise.py`) in the picture of every
controller (default `none`; in `planner_eval.sh` and `selector_eval.sh`, `same` = the depth noise's level).
`INSTRUCTIONS=NAME` gives the 4 tasks the reworded instructions of `instruction_sets.py` instead of their own.
`TASK_SET=harder` runs the 5 harder scenes of `harder_scenes.py` instead of SimplerEnv's 4 tasks (their fixed
layouts only: `LAYOUTS=paper`). `PLAN_CHECK=1` adds the VLM plan check (`plan_check.py`): in `planner_eval.sh` it
only logs the feature `vlm_plan_yes`; in `selector_eval.sh` a "no" gives the episode to the VLA.
`TASK_SOURCE=vlm` replaces the planner's fixed rule with the VLM plan (`vlm_plan.py`; see "VLM plan" below).

**Costs** (since 2026-10-23). Each sim writes `episode_costs.jsonl` (`model_time.py`): for each episode, its
wall-clock seconds from the model's reset to the last step, and its model requests, the count and the round-trip
seconds for each model (`vla`, `vlm`, `sam`). The planner records add what the first look cost (`first_look_seconds`,
`first_look_model`): the part of an episode that the ASM needs in every episode. Every runner logs the GPU every 5 s
in `dmon.txt` (power, use and memory, with timestamps). `score_costs RUN...` gives each eval's cost per episode: the
seconds, the requests and the GPU energy. The GPU energy is the eval's own only when the eval had the GPU to itself:
one eval at a time (`planner_evals.sh` with AT_ONCE=1) and no other user on the GPU. The round trips are GPU time
only with one sim at a time; when sims share the servers (K sim clients, or the planner's 4 tasks at once), they
include the wait for the other sims' requests.

`SAM_VERSION` picks the model of every SAM server, both the VLA's visual prompts and the planner's masks:
`sam3.1` (the default since 2026-10-23: SAM 3.1's detector, from the sam31 env) or `sam3` (transformers'
`Sam3Model`: the paper's setup, and every eval before 2026-10-23). Both run the image encoder once per frame
and keep the last 4 frames, so a frame's second prompt and the planner's retries at lower thresholds reuse it.
Each planner record names the SAM server's model (`sam_model`), and the closed-loop eval uses a saved ASM only
with the SAM model it was trained on. On the paper's test set (3 VLA-only evals, 288 episodes) the two gave the same success rate
(SAM 3 56.6%, SAM 3.1 55.9%, exact McNemar p = 0.85). Examples (the evals before 2026-10-23 ran with
`SAM_VERSION=sam3`; set it to repeat one):

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

The camera-noise study's evals and ASMs (see "Camera-noise study" below). Its none level is the two-model
study's evals, linked under the study's names:

```bash
for set in iid11 iid12; do
  ln -s pseed_${set}_none $RUNS_DIR/cam_${set}_none
  ln -s vla_$set $RUNS_DIR/vla_${set}_none
done
RGB_NOISE=same bash $E/planner_evals.sh 0 10320 2 cam_iid11_realistic:realistic:iid11 cam_iid11_harsh:harsh:iid11 \
    cam_iid12_realistic:realistic:iid12 cam_iid12_harsh:harsh:iid12
for set in iid11 iid12; do
  for level in realistic harsh; do
    RGB_NOISE=$level bash $E/vla_eval.sh 0 10400 vla_${set}_$level $set 4
  done
done
mkdir -p $RUNS_DIR/cam_model
for level in none realistic harsh; do
  python -m examples.SimplerEnv.eval_files.planner.selector --rule two-model \
      --planner-runs $RUNS_DIR/cam_iid11_$level --vla-runs $RUNS_DIR/vla_iid11_$level \
      --save $RUNS_DIR/cam_model/asm_$level.json > $RUNS_DIR/cam_model/train_$level.txt
done
RGB_NOISE=same DEPTHS=harsh CHUNK_SIZE=25 bash $E/selector_eval.sh 0 10510 cam_closed_loop_iid12 iid12 \
    'cam_model/asm_{}.json' 6
```

The reworded-instruction study's evals (see "Reworded-instruction study" below):

```bash
for set in synonyms attributes order reasoning; do
  PLAN_CHECK=1 INSTRUCTIONS=$set bash $E/planner_eval.sh 0 10320 words_${set}_paper none paper
  INSTRUCTIONS=$set bash $E/vla_eval.sh 0 10400 vla_words_${set}_paper paper 4
done
```

The harder-scene study's smoke tests and evals (see "Harder-scene study" below):

```bash
# smoke tests: 2 episodes of each task, the planner given the simulator's object positions (no servers needed)
for t in ClutteredSpoonOnTowel-v0 ClutteredCarrotOnPlate-v0 TwoStepSpoonAndCarrot-v0 CarrotOnNearPlate-v0 \
         CarrotOnFarPlate-v0; do
  bash $E/sim_task.sh 0 $t 0 2 smoke_harder/$t --controller planner --planner mplib --target-source sim
done
# the evals
PLAN_CHECK=1 TASK_SET=harder bash $E/planner_eval.sh 0 10320 harder_paper none paper
TASK_SET=harder bash $E/vla_eval.sh 0 10400 vla_harder_paper paper 4
```

The VLM plan-check study's eval (see "VLM plan-check study" below); it also reads the two studies' evals above:

```bash
PLAN_CHECK=1 bash $E/planner_eval.sh 0 10320 check_paper_none none paper
```

The VLM plan-check v2 study's evals (see "VLM plan-check v2 study" below). The test runs only if the dev step
says go:

```bash
S=examples.SimplerEnv.eval_files.planner.experiments.score_plan_check_v2
export PLAN_CHECK=1 PLAN_CHECK_MARKS=outline,tint
# the dev step: the first study's planner-only evals again, with both checks
bash $E/planner_eval.sh 0 10320 check2_own_paper none paper
for set in synonyms attributes order reasoning; do
  INSTRUCTIONS=$set bash $E/planner_eval.sh 0 10320 check2_words_${set}_paper none paper
done
TASK_SET=harder bash $E/planner_eval.sh 0 10320 check2_harder_paper none paper
python -m $S --dev --save plan_check_v2.json > $RUNS_DIR/plan_check_v2_dev.txt
# the test: planner-only (test 1), then VLA-only (tests 2 to 4)
IID_EPISODES=24 bash $E/planner_eval.sh 0 10320 check2_own_iid13 none iid13
for set in synonyms attributes order reasoning; do
  IID_EPISODES=24 INSTRUCTIONS=$set bash $E/planner_eval.sh 0 10320 check2_words_${set}_iid13 none iid13
done
IID_EPISODES=48 TASK_SET=harder bash $E/planner_eval.sh 0 10320 check2_harder_iid13 none iid13
python -m $S > $RUNS_DIR/plan_check_v2_test1.txt
unset PLAN_CHECK PLAN_CHECK_MARKS
for set in synonyms attributes order reasoning; do
  IID_EPISODES=24 INSTRUCTIONS=$set bash $E/vla_eval.sh 0 10400 vla_words_${set}_iid13 iid13 4
done
IID_EPISODES=48 TASK_SET=harder bash $E/vla_eval.sh 0 10400 vla_harder_iid13 iid13 4
IID_EPISODES=24 bash $E/vla_eval.sh 0 10400 vla_own_iid13 iid13 4  # secondary
python -m $S > $RUNS_DIR/plan_check_v2_test.txt
```

The planner first ran with SAM 3.1 after 2026-10-23, so a smoke test comes before the evals above: 2 episodes
of each task, checked in the records against `pseed_paper_none` (SAM 3, the same episodes): the outcomes, and how
far the camera estimate is from the simulator's object (`grasp_error_xy`, `place_error_xy`). It is not part of
any result. Done on 2026-10-23: the 8 outcomes were the same as with SAM 3 (7 successes), and the estimates were
within 0.8 cm of SAM 3's (at most 3.4 cm from the simulator's object).

```bash
CUDA_VISIBLE_DEVICES=0 $SAM31_PYTHON examples/Robocasa_tabletop/visual_prompt_utility/sam3_server.py \
    --port 10094 --sam-version sam3.1 > sam31_smoke_server.log 2>&1 &
for t in PutSpoonOnTableClothInScene-v0 PutCarrotOnPlateInScene-v0 StackGreenCubeOnYellowCubeBakedTexInScene-v0 \
         PutEggplantInBasketScene-v0; do
  bash $E/sim_task.sh 0 $t 0 2 smoke_sam31/$t --controller planner --planner mplib --target-source camera \
      --sam3-port 10094 --robot-mask joints --planner-seed 0
done
kill %1
```

The learned-features training data, with SAM 3.1 (see "Learned first-frame features" below):

```bash
bash $E/planner_evals.sh 0 10320 2 pseed31_iid11_harsh:harsh:iid11 pseed31_iid14_harsh:harsh:iid14
HIDDEN=1 bash $E/vla_eval.sh 0 10400 vla31_iid11 iid11 4
HIDDEN=1 bash $E/vla_eval.sh 0 10400 vla31_iid14 iid14 4
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
--models 'two_model/asm_{}.json' --vla vla_iid12` (its closed-loop eval). The camera-noise study's: `score_two_model
--runs cam --vla 'vla_{set}_{level}' --noise camera --other` and `check_closed_loop cam_closed_loop_iid12 --planner
'cam_iid12_{}' --models 'cam_model/asm_{}.json' --vla vla_iid12_harsh`. The reworded-instruction study's:
`score_instructions --asm two_model/asm_none.json`. The harder-scene study's: `score_harder_scenes`. The VLM
plan-check study's: `score_plan_check --asm two_model/asm_none.json`. The VLM plan-check v2 study's:
`score_plan_check_v2 --dev --save plan_check_v2.json` (the dev step) and `score_plan_check_v2` (the test). Any
eval's costs (see "Costs" above): `score_costs RUN...`.

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

## Learned first-frame features

The two-model ASM's success models use hand-made features (`selector.PLANNER_FEATURES`, `selector.VLA_FEATURES`).
A one-time CPU look on the two-model study's training set only (`iid11`; `iid12` was not read) asked whether
learned features do better, and whether the ASM can be one model: the first-frame features and the method in,
P(success) out, so that one model gives both P(planner) and P(VLA). Its cross-validation folds held out whole
episode indices, because the 4 tasks of one episode index share a layout (spoon and carrot have the same object
position), and its within-task AUC used only the features' part of the logit, without the task intercepts, which
change from fold to fold. Results, with 95% intervals from a bootstrap over episode indices
(`learned_features_look_2026-10-23.txt`):

- P(VLA): a logistic regression with an L2 penalty on all 2560 numbers of the VLA's last layer (its first-step
  hidden state, `vla_iid11/hidden`) had a within-task AUC of 0.695 against 0.629 for the hand-made features
  (+0.07, -0.02 to +0.16) and a lower log-loss (-0.036, -0.073 to +0.001): a trend, not a significant gain. The
  linear probe (PCA to 10 numbers) did worse than both, and the ground-truth layout (positions and turn from the
  simulator) did worse than the hidden state (0.616). The hand-made features from clean depth, or with a more
  flexible form, did not do better.
- P(planner) with harsh depth: the hidden state is at chance. The planner fails from depth noise, which the
  camera picture does not show.
- Decisions: nothing beat the two-model ASM. The learned P(VLA) and the one-model ASM (a logistic regression or
  a small network) were between 0.9 points below it and 0.2 points above it; the linear probe was 3.6 points below
  (-6.2 to -0.8).
- With 25%, 50%, 75% and all of the about 360 training episodes of a fold, the hidden state's within-task AUC was
  0.617, 0.626, 0.670 and 0.685; the hand-made features' was nearly flat (0.627 with all).

So more training data comes first, before any plan or test set: a second training set (`iid14`) and the training
set again, both with SAM 3.1 (the default), so that all 800 episodes share one setup. The planner-only evals are
at harsh depth only; the VLA-only evals save the first-step hidden states, without the action spread (it added
nothing). Commands: under "Evals (GPU)".

## Camera-noise study

The plan below was fixed on 2026-10-16, before any of the study's evals ran (the RGB noise did not exist before),
and the scorers follow it. The results of the two-model study and of this study do not change it. A result that
is not in the plan is reported as exploratory.

**Question.** In the two-model study, the noise levels change only the planner's depth picture. The VLA uses only
the color picture, which stays perfect, so the comparison favors the VLA. Does the two-model ASM still beat both
methods when both of them get a noisy camera?

**Camera levels.** At camera level L (none, realistic or harsh), the color picture of every controller has RGB
noise L, and the planner's depth picture (only the planner uses depth) has depth noise L, as in the two-model
study. The RGB noise (`camera_noise.py`) is a real color camera's errors:

| Level | Sensor noise at mid-gray | Exposure error (1 sd) | White-balance error (1 sd, per channel) | Blur (sigma) |
|---|---|---|---|---|
| realistic | 4/255 | 5% | 2% | 0.5 px |
| harsh | 11/255 | 16% (a quarter stop) | 5% | 1.5 px |

The exposure, white balance and blur are fixed for an episode (one camera setup); the sensor noise (shot and read
noise) is new in every frame. The noise goes into the picture once per frame, before any controller sees it, so
the VLA's input, its SAM3 visual prompts and its VLM, and the planner's SAM3 masks all see the same pixels. It is
seeded by the noise seed (0) and the episode, so every controller gets the same first frame, and the depth noise's
draws are the same as without it.

**Evals.** Planner-only `cam_iid11_{level}` and `cam_iid12_{level}` (planner seed 0) and VLA-only
`vla_iid11_{level}` and `vla_iid12_{level}` (fast settings, no first-step files) at the realistic and harsh camera
levels; the closed-loop eval `cam_closed_loop_iid12` (harsh camera) with the saved ASMs `cam_model/asm_{level}.json`.
The none level is the two-model study's evals, which have the same settings. The commands are above. About 15 GPU
hours on one GPU (the 4 VLA-only evals about 13 hours).

**Change of 2026-10-23, before any of the study's evals ran.** The study's new evals use SAM 3.1
(`SAM_VERSION=sam3.1`, now the default) for the VLA's visual prompts and the planner's masks. The none level
stays the two-model study's evals, which used SAM 3; it is secondary only and is reported as a SAM 3 result. The
planned tests (harsh camera) use only SAM 3.1 evals and the ASM trained on them.

**Models, rule and tests.** As in the two-model study: one ASM per camera level, trained on `iid11` at that level,
with the same success models, features and decision rule. The planned tests are the same three, in the same fixed
order, on the 400 episodes of `iid12` at the harsh camera level: the two-model ASM against (1) the better single
method on the training set, (2) rule A and (3) rule C. Each is the exact two-sided paired test at 0.05 and counts
only when every test before it was significant; the intervals are as in the two-model study.

**Secondary (reported, no claims).** The none and realistic camera levels; each method's success rate at each
level (how much the camera noise lowers the VLA and the planner); the success models' AUC, within-task AUC,
log-loss, Brier score and calibration; the closed-loop eval against the offline result (`check_closed_loop`). The
action spread and the linear probe are not part of this study.

## Reworded-instruction study

The plan below was fixed on 2026-10-17, before any of the study's evals ran, and `score_instructions` follows it.
The results do not change it. A result that is not in the plan is reported as exploratory.

**Question.** The planner reads an instruction with one fixed rule: it drops a leading verb and "the"; the words
before "on", "onto", "in" or "into" name the object to move, and the words after it name the target (the SAM3
prompts). Then it runs one pick-and-place script. The rule is the same for every task, but it works only when the
instruction has the form "verb object on target", as SimplerEnv's instructions do. Does the planner lose more than
the VLA when the same tasks are given in other words?

**Instruction sets** (`instruction_sets.py`; the scene and the goal stay the same). What the rule reads from each:

| Set | Task | Instruction | Object the rule reads | Target the rule reads |
|---|---|---|---|---|
| own | Spoon | put the spoon on the towel | spoon | towel |
| own | Carrot | put carrot on plate | carrot | plate |
| own | Stack | stack the green block on the yellow block | green block | yellow block |
| own | Eggplant | put eggplant into yellow basket | eggplant | yellow basket |
| synonyms | Spoon | place the spoon on the cloth | spoon | cloth |
| synonyms | Carrot | set the carrot on the dish | set the carrot | dish |
| synonyms | Stack | put the green cube on top of the yellow cube | green cube | top of the yellow cube |
| synonyms | Eggplant | drop the eggplant into the basket | drop the eggplant | basket |
| attributes | Spoon | put the utensil on the towel | utensil | towel |
| attributes | Carrot | put the orange vegetable on the plate | orange vegetable | plate |
| attributes | Stack | stack the green object on the yellow object | green object | yellow object |
| attributes | Eggplant | put the purple vegetable into the basket | purple vegetable | basket |
| order | Spoon | the towel is where the spoon should go | (the whole sentence) | none |
| order | Carrot | the plate should get the carrot | (the whole sentence) | none |
| order | Stack | the yellow block should end up under the green block | (the whole sentence) | none |
| order | Eggplant | the basket is where the eggplant goes | (the whole sentence) | none |
| reasoning | Spoon | put what you would eat soup with on the towel | what you would eat soup with | towel |
| reasoning | Carrot | put the food that rabbits love on the plate | food that rabbits love | plate |
| reasoning | Stack | put the block that is the color of grass on the other block | block that is the color of grass | other block |
| reasoning | Eggplant | put the vegetable that baba ganoush is made from into the basket | vegetable that baba ganoush is made from | basket |

SAM3 takes free text, so the planner can still find an object from a description it was not written for; the
study measures how often. The VLA's VLM reads the instruction before the SAM3 visual prompts are made.

**Evals.** The test set (`paper`, 96 layouts), clean camera. Planner-only `words_{set}_paper` (planner seed 0, as
`pseed_paper_none`) and VLA-only `vla_words_{set}_paper` (fast settings, VLM seed offset 0, as `fast3_train_g0/g1`).
With the tasks' own instructions, `pseed_paper_none` and `fast3_train_g0/g1` are used. The commands are above.
About 3.5 GPU hours on GPU 0. The planner-only evals also run the VLM plan check for the VLM plan-check study
(`PLAN_CHECK=1`, added on 2026-10-17 before any eval ran); it only logs a feature, so the planner's episodes stay
the same.

**Change of 2026-10-23, before any of the study's evals ran.** All the study's evals use SAM 3.1
(`SAM_VERSION=sam3.1`, now the default). The arms with the tasks' own instructions change with them, so that each
test compares evals with the same SAM model: the planner's is `check_paper_none` (planner seed 0, clean camera,
SAM 3.1; the VLM plan-check study's eval, whose check only logs a feature) instead of `pseed_paper_none`, and the
VLA's is `sam31_paper` (VLA-only, SAM 3.1, fast settings, VLM seed offset 0; run on 2026-10-16) instead of
`fast3_train_g0/g1`. Since then the SAM 3.1 code runs the image encoder once per frame instead of once per prompt;
on 156 frames of an eval the two gave the same detections. The secondary ASM (`two_model/asm_none.json`) was
trained on SAM 3 evals, and its result is reported with that label.

**Planned tests.** In this fixed order, each the exact two-sided paired sign-flip test at 0.05, and each counting
only when every test before it was significant. A layout gives one value, its sum over the 4 reworded sets, so
the 4 sets of one layout are not counted as independent episodes.

1. The planner loses more than the VLA: (VLA, reworded minus own instructions) minus (planner, reworded minus own
   instructions).
2. The VLA beats the planner on the reworded instructions: VLA-only minus planner-only.

Intervals: 95% percentile bootstrap of the per-layout mean; each draw takes each task's layouts again, with
replacement (10,000 draws).

**Secondary (reported, no claims).** Each set on its own (exact McNemar: VLA against planner, and each method
against its own result with the tasks' own instructions); the two-model study's ASM for clean depth
(`two_model/asm_none.json`, trained on the tasks' own instructions), offline, from the first look of each
planner-only eval.

## Harder-scene study

The plan below was fixed on 2026-10-17, before any of the study's evals ran, and `score_harder_scenes` follows it.
The results do not change it. A result that is not in the plan is reported as exploratory.

**Question.** The planner reads every instruction with one fixed rule (see "Reworded-instruction study") and then
runs one pick-and-place script. Some tasks cannot be done that way: two pick-and-places asked for in one
instruction, or a target that is named only by where it is. In new scenes with more objects on the table, does the
VLA beat the planner on such tasks, while the planner keeps its lead on the one-step tasks of the same scene?

**Scenes and tasks** (`harder_scenes.py`; `harder_scene_envs.py` builds them in SimplerEnv's simulator). Both scenes
use the table, camera, light and robot of SimplerEnv's spoon and carrot tasks. The kitchen scene has the spoon, the
towel, the carrot and the plate. The two-plate scene has the carrot and two identical plates, one in the row nearer
the robot and one in the far row. What the rule reads from each instruction:

| Task | Scene | Instruction | Object the rule reads | Target the rule reads | Layouts | Steps |
|---|---|---|---|---|---|---|
| Cluttered spoon | kitchen | put the spoon on the towel | spoon | towel | 96 | 120 |
| Cluttered carrot | kitchen | put carrot on plate | carrot | plate | 96 | 120 |
| Two-step | kitchen | put the spoon on the towel and the carrot on the plate | spoon | towel and the carrot on the plate | 96 | 240 |
| Near plate | two plates | put the carrot on the plate that is closer to the camera | carrot | plate that is closer to the camera | 32 | 120 |
| Far plate | two plates | put the carrot on the plate that is farther from the camera | carrot | plate that is farther from the camera | 32 | 120 |

The two cluttered tasks keep the instructions of SimplerEnv's own spoon and carrot tasks, so the rule reads them as
before; only the two extra objects are new. On the other three it fails in a different way. On the two-step task
it reads one pick-and-place, so the planner cannot finish the task: it stops after the first one. On the plate
tasks it gives SAM3 the whole description, and SAM3 finds objects by name, not by where they are, so either plate
can be chosen.

**Layouts.** Each object starts on one corner of a square grid centred where SimplerEnv's layouts are, with
half-edges of 7.5 cm (as in the spoon and carrot tasks) and 10 cm (as in the cube task); objects start at least
2 cm apart. Kitchen scene: each of the 24 ways to put the 4 objects on the 4 corners, on each grid, with each of
the 2 turns of the spoon and the carrot from SimplerEnv's tasks: 96 layouts, the same in its three tasks
(episode N of each has the same objects in the same places). Two-plate scene: one plate on each row, the carrot on
one of the 2 corners left, on each grid, with the carrot task's 2 turns: 32 layouts, the same in both tasks with
the target swapped.

**Success**, checked at the end of the episode as SimplerEnv's tasks check it: the spoon on the towel (partly is
enough, as in the spoon task); the carrot on the plate, touching nothing but the plate and the robot (as in the
carrot task); for the two-step task, both. Each episode also logs each goal, the carrot on the other plate (plate
tasks), and whether an object outside the goals moved more than 3 cm.

**Smoke tests first** (commands above; not part of the results): 2 episodes of each task with the planner given
the simulator's object positions, on GPU 0. Their videos are checked by eye for objects that fall or overlap and
for goals that are logged wrongly, and then deleted. They can change the scenes, but not this plan's tests.

**Evals.** Planner-only `harder_paper` (planner seed 0, clean depth, as `pseed_paper_none`) and VLA-only
`vla_harder_paper` (fast settings, VLM seed offset 0, as `fast3_train_g0/g1`), clean camera, 352 episodes each,
on GPU 0. About 4 GPU hours. The planner-only eval also runs the VLM plan check for the VLM plan-check study
(`PLAN_CHECK=1`, added on 2026-10-17 before any eval ran); it only logs a feature, so the planner's episodes stay
the same.

**Change of 2026-10-23, before any of the study's evals ran.** Both evals use SAM 3.1 (`SAM_VERSION=sam3.1`, now
the default). In the secondary comparison with SimplerEnv's own spoon and carrot tasks, `check_paper_none` and
`sam31_paper` (both SAM 3.1) replace `pseed_paper_none` and `fast3_train_g0/g1`.

**Change of 2026-10-23, after the smoke tests and before any of the study's evals ran.** The smoke tests (on
GPU 1, as the planner-only eval) found no objects that fall or overlap and no goals logged wrongly. (The planner
missed every turned spoon and carrot in them: the simulator's positions carry no turn, so the hand kept its starting
turn. The evals use the camera.) But their videos showed that "the robot" in the plate instructions can mislead: the
robot's base is below the camera picture, while its arm comes in from the top, above the far row. The plate
instructions therefore name the camera instead of the robot (table above). The goals, scenes and tests are the same:
the near row is nearer both the robot's base and the camera (`test_near_plate_is_nearer_the_camera`).

**Planned tests.** In this fixed order, each the exact two-sided paired sign-flip test at 0.05, and each counting
only when every test before it was significant. Each layout gives one value.

1. On the tasks the rule cannot read (two-step, near plate and far plate; 160 layouts): VLA-only minus
   planner-only.
2. In the kitchen scene (96 layouts): the planner's lead over the VLA (planner-only minus VLA-only) on the two
   one-step tasks (their mean) minus its lead on the two-step task.

Intervals: 95% percentile bootstrap of the per-layout mean; each draw takes each task's layouts again, with
replacement (10,000 draws; test 2 has one stratum, the kitchen layouts).

**Secondary (reported, no claims).** Each task on its own (exact McNemar, VLA against planner); the logged episode
stats of each method (each goal of the two-step task, the carrot on the wrong plate, other objects moved); the
two cluttered tasks against SimplerEnv's own spoon and carrot tasks without the extra objects (`pseed_paper_none`
and `fast3_train_g0/g1`; other layouts, so not paired). No ASM is scored here: the saved ASMs have one column per
SimplerEnv task and were not trained on these tasks.

## VLM plan-check study

The plan below was fixed on 2026-10-17, before any of the study's evals ran, and `score_plan_check` follows it.
The results do not change it. A result that is not in the plan is reported as exploratory.

**Question.** The two studies above give the planner tasks its fixed rule cannot read, and the ASMs cannot see
that: their features come from the planner's first look at the scene (did it find a path, how long is it), not
from the instruction. Can the VLM tell, on the first frame, whether the planner's one pick-and-place will finish
the instruction, so that those episodes go to the VLA?

**The check** (`plan_check.py`). On the first frame, after SAM3 has found the object and the target that the rule
reads, the planner marks them on the camera picture: the object it will pick up tinted green, and a blue box
around where it will put it. The VLM (the VP-VLA VLM, Qwen3-VL-4B, with the VLA's fast settings) gets the picture,
the marked picture, the instruction and this question: will picking up the green object, putting it in the blue
box and then stopping complete the whole instruction? Its probability of "yes" is read from its first answer token
(no sampling) and logged as the feature `vlm_plan_yes`. When SAM3 finds no object or no target there is no plan
to show, and the feature is 0. The marked pictures are saved next to the planner records.

**The gate.** The VLA when `vlm_plan_yes` is below 0.5, else the planner. The threshold is fixed here, not fitted:
no training set has tasks that break the rule. The gate needs no task name, so it works on tasks no ASM was
trained on.

**Evals.** The planner-only evals of the two studies above, run with the check (`PLAN_CHECK=1`): `harder_paper`
(352 layouts) and `words_{set}_paper` (4 x 96); with SimplerEnv's own instructions, the new `check_paper_none`
(96 layouts, planner seed 0, clean camera: the same episodes as `pseed_paper_none`, about 0.5 GPU hours on GPU 0).
VLA-only: `vla_harder_paper`, `vla_words_{set}_paper` and `fast3_train_g0/g1`. The gate is scored offline: on each
layout, its choice from the planner-only eval's first frame, and the outcome of the chosen method's eval of that
layout. Both controllers start from the same first frame, so a closed loop should give the same (the two-model
study's closed-loop eval matched its offline scores once the planner was seeded).

**Change of 2026-10-23, before any of the study's evals ran.** All the evals above use SAM 3.1
(`SAM_VERSION=sam3.1`, now the default), so `check_paper_none` is no longer the same episodes as
`pseed_paper_none` (SAM 3), and the VLA-only eval with SimplerEnv's own instructions is `sam31_paper` (SAM 3.1,
fast settings, VLM seed offset 0; see "Reworded-instruction study") instead of `fast3_train_g0/g1`. The
secondary ASM (`two_model/asm_none.json`) was trained on SAM 3 evals, and its result is reported with that label.

**Planned tests.** In this fixed order, each the exact two-sided paired sign-flip test at 0.05, and each counting
only when every test before it was significant. Each layout gives one value; a layout of the reworded
instructions gives its sum over the 4 sets.

1. Harder scenes (352 layouts): the gate minus planner-only.
2. Harder scenes: the gate minus VLA-only.
3. Reworded instructions (96 layouts): the gate minus planner-only.

Intervals: 95% percentile bootstrap of the per-layout mean; each draw takes each task's layouts again, with
replacement (10,000 draws).

**Secondary (reported, no claims).** For each instruction set (own, the 4 reworded sets, the harder scenes): the
gate against planner-only and against VLA-only (exact McNemar), and the AUC of `vlm_plan_yes` for the planner's
success. Per task: the success rates and how many episodes the gate gives the VLA. How much of the VLM's first
token goes to yes or no (near 1 means it answers the question). With `--asm two_model/asm_none.json`: the gate
followed by the two-model study's ASM for clean depth (the ASM decides when the VLM says yes), on SimplerEnv's
own and reworded instructions.

## VLM plan-check v2 study

The plan below was fixed on 2026-10-23, before any of the study's evals ran, and `score_plan_check_v2` follows it.
The results do not change it. A result that is not in the plan is reported as exploratory. The first study ("VLM
plan-check study") is not changed by it and is reported as planned.

**Why.** The first study's planner-only evals (run on 2026-10-23) showed the check saying no to most correct
plans. With SimplerEnv's own instructions the planner finished 90 of 96 episodes, but the VLM said yes on 33 (AUC
of P(yes) for the planner's success 0.45); in the cluttered carrot task the planner finished 96 of 96 and the VLM
said yes on 7; with "orange vegetable" and "purple vegetable" it said yes on none. The marked pictures were right.
A likely cause is the tint: it makes the orange carrot and the purple eggplant look green. As these episodes were
seen before this plan, they are only its dev set.

**The v2 check** (`plan_check.py`, mark `outline`). As the first check, but the object to pick up gets a green
line 3 pixels wide just outside its mask, and its own pixels keep their colour; the question says "outlined in
green" instead of "tinted green". The same VLM (Qwen3-VL-4B-Instruct, the VLA's fast settings) and the same P(yes)
from its first answer token, as the feature `vlm_plan_yes_outline`. Each eval of this study runs both checks on
each first frame (`PLAN_CHECK_MARKS=outline,tint`), so v1 (`vlm_plan_yes`) and v2 are compared on the same
episodes.

**Dev step** (episodes the first study already ran; no test). The first study's planner-only evals again, with
both checks: `check2_own_paper`, `check2_words_{set}_paper` and `check2_harder_paper` (832 episodes; the planner's
episodes are the same as in the first study). On these, once:

- v2's threshold: the P(yes) value t with the largest Youden's J for the planner's success (the planner when
  P(yes) >= t; J = the share of planner successes with a yes + the share of planner failures with a no - 1);
  ties: the t nearest 0.5. Saved in `plan_check_v2.json`.
- Go or no go: the test runs only if v2's AUC of P(yes) for the planner's success, over all 832 episodes, is
  higher than v1's. A no go ends the study without a test, and that is reported.

**Test set** (`iid13`: layout seed 13, a random layout of its own for each episode; no episode of it ran before
this plan). SimplerEnv's 4 tasks, 24 episodes each, with the tasks' own instructions and with each of the 4
reworded sets (5 x 96 episodes; episode N has the same layout in each set). The 5 harder tasks, 48 episodes each
(`harder_scenes.random_layout`: the grid of the fixed layouts with a random half-edge of 7.5 to 9 cm, each object
up to 1 cm off its corner in x and y, the spoon and the carrot turned at random, at least 2 cm between objects;
the 3 kitchen tasks share their layouts, and so do the 2 plate tasks with the target swapped): 240 episodes. 720
planner-only episodes in all.

**Evals.** Planner-only (planner seed 0, clean depth, SAM 3.1, both checks): `check2_own_iid13`,
`check2_words_{set}_iid13` and `check2_harder_iid13`. VLA-only (fast settings, VLM seed offset 0):
`vla_words_{set}_iid13` and `vla_harder_iid13`, and `vla_own_iid13` for the secondary results. About 1 GPU hour
for the dev step and the planner-only test evals, and 5 to 7 GPU hours for the VLA-only evals. The commands are
above.

**Planned tests.** In this fixed order, each two-sided at 0.05, and each counting only when every test before it
was significant.

1. v2 predicts the planner's success better than v1: on the 720 planner-only test episodes, the AUC of v2's P(yes)
   minus v1's, by DeLong's test for two AUCs on the same episodes; significant only with v2's AUC the higher.
   Interval: 95% percentile bootstrap of the AUC difference; each draw takes each task's episodes of each
   instruction set again, with replacement (10,000 draws). It needs no VLA-only eval.
2. Harder scenes (240 layouts): the v2 gate (the VLA when v2's P(yes) is below the dev step's threshold, else the
   planner) minus planner-only.
3. Harder scenes: the v2 gate minus VLA-only.
4. Reworded instructions (96 layouts; a layout's value is its sum over the 4 sets): the v2 gate minus planner-only.

Tests 2 to 4 are scored offline as in the first study (the gate's choice from the planner-only eval's first frame,
and the outcome of the chosen method's eval of that layout), each by the exact paired sign-flip test with one value
per layout, with intervals as in the first study.

**Secondary (reported, no claims).** For each instruction set: v1's and v2's AUC and mean within-task AUC, how
often each says yes, and v2's probability on yes/no as its first word; with the VLA-only evals: planner-only,
VLA-only, the v2 gate (exact McNemar against each), the v1 gate at 0.5, the oracle, and how many episodes v2 gives
the VLA, also with the tasks' own instructions (`vla_own_iid13`). The dev step's tables (exploratory).

## VLM plan

A method, not yet a study: its study plan is written here before any eval with it runs.

The planner's fixed rule reads one object and one target from an instruction of the form "put X on Y", and the
planner then does one pick-and-place. With `TASK_SOURCE=vlm` (`--task-source vlm`) the VLM writes the planner's
job instead (`vlm_plan.py`). On the first frame it gets the picture and the instruction and answers whether the
instruction can be done with pick-and-place moves only; if so it lists the moves in order (at most 4), each an object
and a target with a short name for SAM3 and a point in the picture (greedy decoding, so the same frame gives the same
plan). The planner does the moves one after another: after each release it rises 6 cm, looks again and plans the
next move. SAM3 finds each name as before; when it finds several masks, the one that covers the VLM's point is used
(else the nearest), which tells two plates apart. A plan the VLM finds not to be pick-and-place ends a planner-only
episode; in the selector mode it gives the episode to the VLA before the ASM decides. An ASM is used only with
records of the same task source (`task_source` is a run setting).

**Why a VLM.** The SAM 3 paper (Carion et al., "SAM 3: Segment Anything with Concepts", arXiv:2511.16719) limits
SAM 3 to short noun phrases with one or two attributes: it is not made for long referring expressions or for
queries that need reasoning. For those, the paper lets a multimodal LLM write the noun phrases and check SAM 3's
masks (its "SAM 3 Agent"). The fixed rule gives SAM parts of the instruction as they are, so reworded and harder
instructions also reach SAM in a form it was not made for. The VLM plan gives SAM short names, and the VLM's
points do the spatial part ("the plate closer to the camera") that SAM does not do.

**SAM score check** (`VLM_PLAN_MIN_SCORE`, `--vlm-plan-min-score`; default 0: no check). The score of a SAM mask
is its match score times its presence score (is the name in the picture at all). A name whose chosen mask scores
below the minimum counts as not found; on the first frame this ends the planner's try (in the selector mode the
VLA gets the episode). The SAM 3 paper counts a detection above 0.5, but correct objects in the simulator often
score far lower (SAM 3: the spoon about 0.35, one eggplant 0.04; SAM 3.1 in the smoke test: the spoon 0.07), so
0.5 would reject them. The minimum is therefore not fixed here: the VLM-plan study's plan fixes it from
its training set, before any test eval runs.

**Fixed for the VLM-plan study on 2026-10-23, before any eval with the VLM plan ran** (the rest of its plan is
written before its evals run too): its evals use SAM 3.1, and the ASM's candidate planner features are
`plan_ok`, `plan_path_ratio` and the first move's SAM scores, `grasp_sam3_score` and `place_sam3_score` (the best
score of the object's and the target's name on the first frame). In SimplerEnv's own scenes the SAM scores carried
no signal, because SAM always found the objects; with the harder scenes and the VLM's names that may change.
Which candidates the ASM uses is chosen on the training set only.

The records add `task_source`, `vlm_plan` (the parsed plan), `vlm_plan_text` (the VLM's answer),
`vlm_plan_feasible`, `steps_done`, `vlm_plan_min_score` and `vlm_plan_sam` (for each name looked up: the move, object
or target, the name, its SAM score, whether the chosen mask covers the VLM's point, and whether the score check
rejected it). The robot-contact check (`transit_contact`) covers the first move only.
