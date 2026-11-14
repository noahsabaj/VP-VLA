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
- `OFF_LIMITS_GPUS` (optional, Linux): nvidia-smi's indices of the GPUs that no process of a runner may open, with
  commas (`1`, `1,3`). Set it on a machine where a GPU is not yours to use; see "Off-limits GPUs".

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
| VLM-plan study: test set | `--layout-seed 15 --layout-per-episode` | `iid15` | 400 (100 per task) |
| marriage study: T1 | `--layout-seed 16 --layout-per-episode` | `iid16` | 400 (100 per task) |
| marriage study: T3 | `--layout-seed 17 --layout-per-episode` | `iid17` | harder scenes 240 (48 per task) |
| marriage study: the demonstrations' pool | `--layout-seed 100 --layout-per-episode` | `iid100` | planner 2,800 (700 per task); VLA-0 until 250 successes per task |
| marriage study: the corrections | `--layout-seed 101 --layout-per-episode` | `iid101` | until 250 successful corrections per task |

In the first three sets, a task's episodes share a few turns of the object: SimplerEnv's layouts have 1 to 3
turns per task, and a `--layout-seed` set has 12 placements per task, each with the same 4 turns. In the `iidN`
sets, each episode has a random placement and turn of its own (`IID_EPISODES` sets the episodes per task). The
harder scenes take `paper` (their fixed layouts) or `iidN` (`harder_scenes.random_layout`).

Planner-only evals are named `{prefix}_{layouts}_{level}`, with the depth-noise level `none` (clean),
`realistic` or `harsh`, or `harsh_xS` (harsh's errors times S; see "Difficulty sweep"). The current ones are `pseed_*` (planner seeded per episode). The VLA-only evals:
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
`vla31_{iid11,iid14}` (SAM 3.1; see "Learned first-frame features"). The VLM-plan study's are
`vlmplan_{iid11,iid15}_{none,harsh}` (planner-only with the VLM plan), `vla31_iid15`, `pseed31_iid15_harsh` and the
closed-loop eval `closed_loop_vlmplan_iid15`, with the ASMs `vlmplan_model/asm_{level}.json`, and the role split's
closed-loop eval `handoff_vlmplan_iid15` (see "VLM-plan study"). The difficulty sweep adds `vlmplan_iid15_harsh_xS`
(see "Difficulty sweep"), and the missing-depth study `miss_{iid11,iid15}_none` (see "Missing-depth study"). The
harder-task system study's are `vlmplan_harder_paper_harsh`, `vlmplan_{own,words_SET,harder}_iid13_harsh` and
`pseed31_{own,words_SET,harder}_iid13_harsh` (see "Harder-task system study"). The marriage study's are named in
its plan (see "Marriage study"): the demonstrations `demos_*`, the trained models' VLA-only evals `{vlap, vlapl,
vlaplt, vlapltd, vlas}_*`, VLA-0's new ones `vla31_iid16` and `vla_harder_iid17`, and the planner-only evals
`pseed31_*` (the fixed rule) and `vp_*` (VP-VLA's VLM marks).

## Evals (GPU)

| Runner | What it runs |
|---|---|
| `planner_eval.sh GPU PORT NAME LEVEL LAYOUTS` | one planner-only eval (one SAM server; the 4 tasks at the same time) |
| `planner_evals.sh GPU PORT_BASE AT_ONCE NAME:LEVEL:LAYOUTS...` | several planner-only evals at the same time |
| `vla_eval.sh GPU PORT_BASE NAME LAYOUTS K [TASK:START:END...]` | one VLA-only eval (one server set, K sim clients) |
| `selector_eval.sh GPU PORT_BASE NAME LAYOUTS MODELS K [LEVEL:TASK:START:END...]` | the closed-loop ASM eval (`CONTROLLER=planner_then_vla`: the role split; `vla_then_planner`: the planner's corrections) |
| `sim_task.sh GPU TASK START END OUT [ARGS...]` | one sim run (smoke tests; the servers must already run) |
| `frames_eval.sh GPU NAME LAYOUTS` | the first frame of every episode of a layout set (the frontier-model study's plan arm; no model server) |
| `agent_eval.sh GPU NAME LAYOUTS [TASK:START:END...]` | one eval of the frontier agent (the frontier-model study's agent arm; the model answers through the relay) |

Each script's header lists its options. The VLA runners use the fast settings by default (`FAST=1`, see
`fast_defaults` in `common.sh`; `FAST=0`: the paper's settings) and an MPS daemon of their own (`MPS=1`). A
run's `status.log` has a START line with its exact settings and commit, one line per chunk or task, and a DONE
line. `RGB_NOISE=realistic|harsh` puts a real color camera's errors (`camera_noise.py`) in the picture of every
controller (default `none`; in `planner_eval.sh` and `selector_eval.sh`, `same` = the depth noise's level).
`INSTRUCTIONS=NAME` gives the 4 tasks the reworded instructions of `instruction_sets.py` instead of their own.
`TASK_SET=harder` runs the 5 harder scenes of `harder_scenes.py` instead of SimplerEnv's 4 tasks (their fixed
layouts, `LAYOUTS=paper`, or a random one per episode, `LAYOUTS=iidN`), `TASK_SET=novel` the 8 new test scenes of
`novel_scenes.py` and `TASK_SET=choice` the 4 choice scenes of `choice_scenes.py` (both with a random layout per
episode only: `LAYOUTS=iidN`). `PLAN_CHECK=1` adds the VLM plan check (`plan_check.py`): in `planner_eval.sh` it
only logs the feature `vlm_plan_yes`; in `selector_eval.sh` a "no" gives the episode to the VLA.
`TASK_SOURCE=vlm` replaces the planner's fixed rule with the VLM plan (`vlm_plan.py`; see "VLM plan" below), and
`TASK_SOURCE=vp` with VP-VLA's own VLM marks (see "VP-VLA's VLM marks" below). With `TASK_SOURCE=vlm`,
`VLM_PLAN_ANSWERS=FILE` reads another model's saved answers to the VLM plan's question in place of the VLM server
(see "Frontier-model study" below). `MISSING_DEPTH=target` takes all depth away from the place target in half of the episodes (see "Missing-depth
study"). In `selector_eval.sh`, `CONTROLLER=planner_then_vla` runs the role split instead of the ASM: the planner
moves the hand to above the object, and then the VLA does the rest of the episode (MODELS must be `-`).
`CONTROLLER=vla_then_planner` records the planner's corrections of the VLA (see "Corrections" below).

**Demonstrations** (since 2026-10-30). `RECORD_DEMOS=1` records every episode of any runner as a demonstration for
the VLA, in each sim's folder (`.../demos`; `demos.py`): the camera pictures, each step's command as the VLA's 7
action numbers (the exact inverse of the VLA's own conversion), the hand's pose, the instruction, which of the
episode's objects each pixel shows (the simulator's own segmentation) and where each object is.
`python -m examples.SimplerEnv.train_files.planner_demos_to_lerobot --out DATA_ROOT/NAME --stats
CHECKPOINT_RUN/dataset_statistics.json RUN...` makes a VP-VLA training set (starVLA's LeRobot v2.0 layout, the
Bridge data config's keys) of the successful ones, and prints how many actions training would clip. With
`--visual-prompts VP_DIR` it also writes the visual prompts that VP-VLA's training draws (a crosshair on the object to
move, a box around its place) from the simulator's masks instead of SAM's, with the subtasks taken from the
recording instead of a VLM (a subtask with "on", "onto", "upon", "in", "into" or "to" puts something down and so
names the place; the tests check that every training rewording gets its box), and prints how often each task's
steps show them. A scene with several goals (the harder scenes' two-step task) needs an instruction with one part for
each goal, and its prompts take the goals in the order they were done, each until the release that put its object in
place (from where the objects were). The eggplant's basket is part of the
background picture, so its box is around SimplerEnv's invisible target plane in the basket: check it in the first
recorded demonstrations. With `--reword` each episode is also written once with a training rewording of each of the
test sets' four families (`train_files/rewordings.py`, 8 per task and family, for SimplerEnv's 4 tasks and the 5
harder scenes). Their exclusion list of every test instruction (the four reworded sets and the harder scenes; since
its change of 2026-10-30, the new test scenes instead of the harder scenes) and of the words that carry each test
family's concepts was fixed on 2026-10-30 before any rewording was written; the tests check every rewording against
it. `train_files/select_demos.py` picks the marriage study's episodes for each model from its pools and writes the
model's set (see "Marriage study").

**Fine-tuning** (since 2026-10-30). `DATA_ROOT=... VP_DIR=... bash examples/SimplerEnv/train_files/run_finetune_1gpu.sh
RUN_ID DATA_MIX` fine-tunes the released VP-VLA checkpoint on one 48 GB GPU with VP-VLA's own training code. The
study's mixtures (`starVLA/dataloader/gr00t_lerobot/mixtures.py`) have one set per model: `vla_p` (`planner_demos`),
`vla_pl` (`planner_demos_reworded`), `vla_plt` (`targeted_demos_reworded`) and `vla_s` (`vla_successes_reworded`),
each written by `planner_demos_to_lerobot.py --out DATA_ROOT/SET --visual-prompts VP_DIR`. Every setting that the
script does not name is the released run's (its `config.yaml`): the model, which is Qwen3-VL-4B with a small MLP
action head that reads the VLM's 16 action tokens (not a DiT), both camera pictures (with and without the visual
prompts), the learning rates and their schedule, and the visual-prompt co-training loss. Changed for one GPU: the
vision encoder, the word embeddings and the lowest 24 of the 36 language-model layers stay frozen, so 1.28 of the
4.50 billion weights train; one process takes 16 samples per step, against 8 GPUs with 32 each; and the co-training
examples come from the training batch, as the recorded sets have no extracted frames. The checkpoint loads module by
module with strict key checks, so a model that does not match it stops the run. The run folder has the layout that
the evals read (`CKPT=RUN_ROOT/RUN_ID/checkpoints/steps_N_pytorch_model.pt`). `DRY_RUN=1` checks the files and prints
the command. The steps, the batch, the learning rates and the frozen layers are settings (MAX_STEPS, BATCH, LR_ACTION,
LR_VLM, FROZEN_LAYERS): a GPU smoke test measures the memory and the speed first, and the study's plan then fixes them
for every model.

**Corrections** (since 2026-10-30). `CONTROLLER=vla_then_planner TAKEOVER_STEPS="LO HI" RECORD_DEMOS=1 DEPTHS=none
bash examples/SimplerEnv/eval_files/planner/experiments/selector_eval.sh GPU PORT_BASE NAME LAYOUTS - K` records the
planner's corrections of the VLA, the study's DAgger data (`planner_client.py`, "VLA then planner"). In each episode
the VLA drives first, for a number of steps drawn from LO to HI (per episode, from the planner seed and the episode
key, so a rerun takes over at the same step), and then the planner takes over from wherever the VLA left the arm. If
the fingers hold the object (the VLA's last command closed them, and they did not close on nothing), the planner lifts
the object and places it, as after its own grasp; otherwise it opens the hand and does the whole pick-and-place, as in
the planner-only mode. If the simulator already counts the task as done at the takeover step, the planner does not
take over and the VLA goes on. Each recording keeps only the planner's part (`first_step` in its `meta.json` is the
takeover step), so the training set learns only the planner's actions, from the states that the VLA reached; an
episode without a takeover is recorded with no steps, and `planner_demos_to_lerobot.py` leaves it out. A recording
that starts with the object in the hand starts its subtasks after the pick part. Each episode's planner record has
`takeover_step`, `takeover` (`holding`, `grasp`, `skipped`, or null when the episode ended first) and `takeover_grip`
(the finger opening at the takeover). The planner's part must fit in the episode's 120 steps, so HI should leave it
about 80 steps; the study's plan fixes LO and HI. The corrections take the fixed rule's one pick-and-place only (no
TASK_SOURCE or PLAN_CHECK).

**Costs** (since 2026-10-23). Each sim writes `episode_costs.jsonl` (`model_time.py`): for each episode, its
wall-clock seconds from the model's reset to the last step, and its model requests, the count and the round-trip
seconds for each model (`vla`, `vlm`, `sam`). The planner records add what the first look cost (`first_look_seconds`,
`first_look_model`): the part of an episode that the ASM needs in every episode. Every runner logs the GPU every 5 s
in `dmon.txt` (power, use and memory, with timestamps). `score_costs RUN...` gives each eval's cost per episode: the
seconds, the requests and the GPU energy. The GPU energy is the eval's own only when the eval had the GPU to itself:
one eval at a time (`planner_evals.sh` with AT_ONCE=1) and no other user on the GPU. The round trips are GPU time
only with one sim at a time; when sims share the servers (K sim clients, or the planner's 4 tasks at once), they
include the wait for the other sims' requests. Since 2026-10-30 the records also have the controller's own seconds
(`controller`: the wall-clock and CPU seconds in the model's reset and steps). `ENERGY=1` makes an energy run (see
"Energy study"; one sim at a time): each episode's GPU energy from the GPU's own energy counter (`gpu_joules`), the
GPU's idle power with the eval's servers loaded (`idle_power.json`), and the same episodes with a controller that
does nothing (`RUN_null`: the simulation alone). Since 2026-11-13 the records also have the simulator's episode stats
after the last step (`episode_stats`; in the harder, new and choice scenes: each goal and each logged pair, e.g.
`spoon_on_plate`, and `moved_other_obj`), so a VLA-only eval has them too, not only in its video names.

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

The VLM-plan study's evals and ASMs (see "VLM-plan study" below), one eval at a time, so that each eval's GPU energy
is its own. Its training set's VLA-only eval (`vla31_iid11`) and fixed-rule eval (`pseed31_iid11_harsh`) are the
learned-features data above:

```bash
export TASK_SOURCE=vlm VLM_PLAN_MIN_SCORE=0
bash $E/planner_evals.sh 0 10320 1 vlmplan_iid11_none:none:iid11 vlmplan_iid11_harsh:harsh:iid11
mkdir -p $RUNS_DIR/vlmplan_model
for level in none harsh; do
  python -m examples.SimplerEnv.eval_files.planner.selector --rule two-model \
      --planner-runs $RUNS_DIR/vlmplan_iid11_$level --vla-runs $RUNS_DIR/vla31_iid11 \
      --planner-candidates plan_ok,plan_path_ratio plan_ok,plan_path_ratio,grasp_sam3_score,place_sam3_score \
      --save $RUNS_DIR/vlmplan_model/asm_$level.json > $RUNS_DIR/vlmplan_model/train_$level.txt
done
# the test set, once the ASMs are saved
bash $E/planner_evals.sh 0 10320 1 vlmplan_iid15_none:none:iid15 vlmplan_iid15_harsh:harsh:iid15
DEPTHS=harsh CHUNK_SIZE=25 bash $E/selector_eval.sh 0 10510 closed_loop_vlmplan_iid15 iid15 \
    'vlmplan_model/asm_{}.json' 6
unset TASK_SOURCE VLM_PLAN_MIN_SCORE
bash $E/vla_eval.sh 0 10400 vla31_iid15 iid15 4
bash $E/planner_evals.sh 0 10320 1 pseed31_iid15_harsh:harsh:iid15
```

The evals of the studies planned on 2026-10-25 (see "VLM-plan study", "Difficulty sweep", "Missing-depth study"
and "Harder-task system study" below), each alone on GPU 0. They also read the VLM-plan study's evals above. The
harder-task system study needs the VLA-only evals of the harder-scene study (`vla_harder_paper`) and of the
plan-check v2 study (`vla_{own,words_SET,harder}_iid13`; commands above), whatever that study's dev step says:

```bash
export TASK_SOURCE=vlm VLM_PLAN_MIN_SCORE=0
# the VLM-plan study's role split: the planner to above the object, then the VLA (closed loop, no ASM)
CONTROLLER=planner_then_vla DEPTHS="harsh none" CHUNK_SIZE=25 bash $E/selector_eval.sh 0 10510 \
    handoff_vlmplan_iid15 iid15 - 6
# the difficulty sweep
bash $E/planner_evals.sh 0 10320 1 vlmplan_iid15_harsh_x0.25:harsh_x0.25:iid15 \
    vlmplan_iid15_harsh_x0.5:harsh_x0.5:iid15 vlmplan_iid15_harsh_x0.75:harsh_x0.75:iid15 \
    vlmplan_iid15_harsh_x1.5:harsh_x1.5:iid15 vlmplan_iid15_harsh_x2:harsh_x2:iid15
# the missing-depth study
MISSING_DEPTH=target bash $E/planner_evals.sh 0 10320 1 miss_iid11_none:none:iid11 miss_iid15_none:none:iid15
# the harder-task system study: the harder scenes' training eval, then the test set
TASK_SET=harder bash $E/planner_eval.sh 0 10320 vlmplan_harder_paper_harsh harsh paper
IID_EPISODES=24 bash $E/planner_eval.sh 0 10320 vlmplan_own_iid13_harsh harsh iid13
for set in synonyms attributes order reasoning; do
  IID_EPISODES=24 INSTRUCTIONS=$set bash $E/planner_eval.sh 0 10320 vlmplan_words_${set}_iid13_harsh harsh iid13
done
IID_EPISODES=48 TASK_SET=harder bash $E/planner_eval.sh 0 10320 vlmplan_harder_iid13_harsh harsh iid13
unset TASK_SOURCE VLM_PLAN_MIN_SCORE
# the harder-task system study's fixed rule (secondary)
IID_EPISODES=24 bash $E/planner_eval.sh 0 10320 pseed31_own_iid13_harsh harsh iid13
for set in synonyms attributes order reasoning; do
  IID_EPISODES=24 INSTRUCTIONS=$set bash $E/planner_eval.sh 0 10320 pseed31_words_${set}_iid13_harsh harsh iid13
done
IID_EPISODES=48 TASK_SET=harder bash $E/planner_eval.sh 0 10320 pseed31_harder_iid13_harsh harsh iid13
```

The energy study's ASMs and energy runs (see "Energy study" below), each run alone on GPU 0. The ASMs need
`vla31_iid11` and `pseed31_iid11_harsh` (the VLM-plan study's) and are saved before any energy run:

```bash
bash $E/planner_evals.sh 0 10320 1 pseed31_iid11_none:none:iid11
mkdir -p $RUNS_DIR/energy_model
for level in none harsh; do
  python -m examples.SimplerEnv.eval_files.planner.selector --rule two-model \
      --planner-runs $RUNS_DIR/pseed31_iid11_$level --vla-runs $RUNS_DIR/vla31_iid11 \
      --save $RUNS_DIR/energy_model/asm_$level.json > $RUNS_DIR/energy_model/train_$level.txt
done
# the energy runs: the first 16 episodes of each task of iid15, one sim at a time
export ENERGY=1 IID_EPISODES=16
bash $E/vla_eval.sh 0 10400 energy_vla_iid15 iid15 1
TASKS_AT_ONCE=1 bash $E/planner_eval.sh 0 10320 energy_pseed31_iid15_harsh harsh iid15
TASKS_AT_ONCE=1 bash $E/planner_eval.sh 0 10320 energy_pseed31_iid15_none none iid15
DEPTHS="harsh none" bash $E/selector_eval.sh 0 10510 energy_closed_loop_iid15 iid15 'energy_model/asm_{}.json' 1
unset ENERGY IID_EPISODES
```

The VLM-router study's eval (see "VLM-router study" below): the fixed rule's planner-only eval of `iid14` again, with
the VLM router's questions on each first frame. Its VLA-only eval, `vla31_iid14`, is the learned-features data above:

```bash
VLM_ROUTER=1 bash $E/planner_eval.sh 0 10320 router_iid14_harsh harsh iid14
```

The marriage study's runs (see "Marriage study" below). `RUN_ROOT` is the
fine-tuning runs' folder and `STEPS` their steps (fixed after the smoke tests); `CHECKPOINT_RUN` is the released
checkpoint's run folder, and `DATA_ROOT` and `VP_DIR` are the folders of the training sets and of their visual
prompts:

```bash
# the smoke tests (not part of the results): the pool's first 20 episodes of each task, converted, then a short
# training run on them
IID_EPISODES=20 RECORD_DEMOS=1 bash $E/planner_eval.sh 0 10320 smoke_demos_iid100 none iid100
python -m examples.SimplerEnv.train_files.planner_demos_to_lerobot --out $SMOKE_ROOT/planner_demos \
    --stats $CHECKPOINT_RUN/dataset_statistics.json --visual-prompts $SMOKE_VP $RUNS_DIR/smoke_demos_iid100
DATA_ROOT=$SMOKE_ROOT VP_DIR=$SMOKE_VP MAX_STEPS=200 WARMUP=20 \
    bash examples/SimplerEnv/train_files/run_finetune_1gpu.sh smoke_vla_p vla_p
CUDA_VISIBLE_DEVICES=0 python -m examples.SimplerEnv.train_files.small_policy train \
    --config $CHECKPOINT_RUN/config.yaml --data-root $SMOKE_ROOT --visual-prompts $SMOKE_VP --out $SMOKE_ROOT/small \
    --steps 200 --warmup 20
# the demonstrations: the planner on the pool's episodes 0 to 699 of each task, VLA-0 on 0 to 399 (then 100 more at
# a time, as explicit chunks, for a task with fewer than 250 successes)
IID_EPISODES=700 RECORD_DEMOS=1 bash $E/planner_eval.sh 0 10320 demos_planner_iid100 none iid100
IID_EPISODES=400 RECORD_DEMOS=1 bash $E/vla_eval.sh 0 10400 demos_vla_iid100 iid100 4
# the planner-only evals of the test sets and of the ASMs' training sets
bash $E/planner_evals.sh 0 10320 1 pseed31_iid16_none:none:iid16 pseed31_iid16_realistic:realistic:iid16 \
    pseed31_iid16_harsh:harsh:iid16
IID_EPISODES=48 TASK_SET=harder bash $E/planner_eval.sh 0 10320 pseed31_harder_iid17_none none iid17
TASK_SET=harder bash $E/planner_eval.sh 0 10320 pseed31_harder_paper_none none paper
export TASK_SOURCE=vp
bash $E/planner_evals.sh 0 10320 1 vp_iid16_none:none:iid16 vp_iid16_realistic:realistic:iid16 \
    vp_iid16_harsh:harsh:iid16 vp_iid11_none:none:iid11 vp_iid11_harsh:harsh:iid11
for set in synonyms attributes order reasoning; do
  INSTRUCTIONS=$set bash $E/planner_eval.sh 0 10320 vp_words_${set}_paper_none none paper
done
IID_EPISODES=48 TASK_SET=harder bash $E/planner_eval.sh 0 10320 vp_harder_iid17_none none iid17
TASK_SET=harder bash $E/planner_eval.sh 0 10320 vp_harder_paper_none none paper
unset TASK_SOURCE
# VLA-0 on T1 and T3 (its T2 evals are the reworded-instruction study's)
bash $E/vla_eval.sh 0 10400 vla31_iid16 iid16 4
IID_EPISODES=48 TASK_SET=harder bash $E/vla_eval.sh 0 10400 vla_harder_iid17 iid17 4
# each trained model on T1 and T2; VLA-PL also on T3 and on the ASMs' training sets
for M in vlap vlapl vlaplt vlapltd vlas; do
  export CKPT=$RUN_ROOT/$M/checkpoints/steps_${STEPS}_pytorch_model.pt
  bash $E/vla_eval.sh 0 10400 ${M}_iid16 iid16 4
  for set in synonyms attributes order reasoning; do
    INSTRUCTIONS=$set bash $E/vla_eval.sh 0 10400 ${M}_words_${set}_paper paper 4
  done
done
export CKPT=$RUN_ROOT/vlapl/checkpoints/steps_${STEPS}_pytorch_model.pt
IID_EPISODES=48 TASK_SET=harder bash $E/vla_eval.sh 0 10400 vlapl_harder_iid17 iid17 4
bash $E/vla_eval.sh 0 10400 vlapl_iid11 iid11 4
TASK_SET=harder bash $E/vla_eval.sh 0 10400 vlapl_harder_paper paper 4
# the corrections: VLA-PLT drives, then the planner takes over (then 100 more episodes at a time, as explicit
# chunks, for a task with fewer than 250 successful corrections)
export CKPT=$RUN_ROOT/vlaplt/checkpoints/steps_${STEPS}_pytorch_model.pt
CONTROLLER=vla_then_planner TAKEOVER_STEPS="10 40" RECORD_DEMOS=1 DEPTHS=none \
    bash $E/selector_eval.sh 0 10510 demos_corrections_iid101 iid101 - 6
unset CKPT
# the blank and wrong instructions (instruction_sets.py): VLA-0 and VLA-PL on T1's first 24 episodes of each task
for set in blank wrong; do
  IID_EPISODES=24 INSTRUCTIONS=$set bash $E/vla_eval.sh 0 10400 vla31_${set}_iid16 iid16 4
  CKPT=$RUN_ROOT/vlapl/checkpoints/steps_${STEPS}_pytorch_model.pt IID_EPISODES=24 INSTRUCTIONS=$set \
      bash $E/vla_eval.sh 0 10400 vlapl_${set}_iid16 iid16 4
done
# VLA-0 reading the subtask on T2
for set in synonyms attributes order reasoning; do
  ACTION_INSTRUCTION=subtask INSTRUCTIONS=$set bash $E/vla_eval.sh 0 10400 vlasub_words_${set}_paper paper 4
done
# the change of 2026-10-30: the harder scenes' demonstrations (the planner reading the VLM's marks; VLA-0, then 100
# more episodes at a time, as explicit chunks, up to 600 of a scene)
IID_EPISODES=700 RECORD_DEMOS=1 TASK_SET=harder TASK_SOURCE=vp \
    bash $E/planner_eval.sh 0 10320 demos_vp_harder_iid100 none iid100
IID_EPISODES=400 RECORD_DEMOS=1 TASK_SET=harder bash $E/vla_eval.sh 0 10400 demos_vla_harder_iid100 iid100 4
# the selection report (train_files/select_demos.py): each pool's episodes, what each task needs, and the chunks still
# to run as the runners' arguments (planner_eval.sh, vla_eval.sh and selector_eval.sh take chunks after their other
# arguments), e.g. 300 more episodes of a harder scene for the planner
python -m examples.SimplerEnv.train_files.select_demos --runs $RUNS_DIR
RECORD_DEMOS=1 TASK_SET=harder TASK_SOURCE=vp \
    bash $E/planner_eval.sh 0 10320 demos_vp_harder_iid100 none iid100 TwoStepSpoonAndCarrot-v0:700:1000
# each training set once the report says that its pools are done, and its model (before the models' evals above);
# VLA-PLTD's set needs the corrections above. The learning curve's models have the T1 evals only.
SD="python -m examples.SimplerEnv.train_files.select_demos --runs $RUNS_DIR --data-root $DATA_ROOT \
    --visual-prompts $VP_DIR --stats $CHECKPOINT_RUN/dataset_statistics.json"
for job in vlap:vla_p:planner_demos vlapl:vla_pl:planner_demos_reworded vlaplt:vla_plt:targeted_demos_reworded \
    vlas:vla_s:vla_successes_reworded vlap450:vla_p450:planner_demos_450 vlap1125:vla_p1125:planner_demos_1125; do
  IFS=: read -r M MIX SET <<< "$job"
  $SD --write $SET && DATA_ROOT=$DATA_ROOT VP_DIR=$VP_DIR MAX_STEPS=$STEPS \
      bash examples/SimplerEnv/train_files/run_finetune_1gpu.sh $M $MIX
done
$SD --write targeted_corrections_reworded && DATA_ROOT=$DATA_ROOT VP_DIR=$VP_DIR MAX_STEPS=$STEPS \
    bash examples/SimplerEnv/train_files/run_finetune_1gpu.sh vlapltd vla_pltd
for M in vlap450 vlap1125; do
  CKPT=$RUN_ROOT/$M/checkpoints/steps_${STEPS}_pytorch_model.pt bash $E/vla_eval.sh 0 10400 ${M}_iid16 iid16 4
done
# the small policy: trained on VLA-P's set (written above), then evaluated on T1 with VP-VLA's visual prompts
CUDA_VISIBLE_DEVICES=0 python -m examples.SimplerEnv.train_files.small_policy train \
    --config $CHECKPOINT_RUN/config.yaml --data-root $DATA_ROOT --visual-prompts $VP_DIR --out $RUN_ROOT/small
POLICY=small CKPT=$RUN_ROOT/small/checkpoints/steps_20000_pytorch_model.pt bash $E/vla_eval.sh 0 10400 small_iid16 iid16 4
# T5's smoke tests (not part of the results): the planner reading the VLM's marks, and reading the simulator's
# positions on the one-goal tasks
IID_EPISODES=8 TASK_SET=novel TASK_SOURCE=vp bash $E/planner_eval.sh 0 10320 smoke_vp_novel_iid98 none iid98
for t in NovelSpongeOnPlate-v0 NovelCanOnTowel-v0 NovelEggplantOnPlate-v0 NovelCarrotInBasket-v0 \
    NovelNotOrangeOnTowel-v0 NovelCarrotByTowelOnPlate-v0; do
  bash $E/sim_task.sh 0 $t 0 8 smoke_sim_novel_iid98/$t --controller planner --planner mplib --target-source sim \
      --robot-mask joints --noise-seed 0 --layout-seed 98 --layout-per-episode
done
# the same with the smaller sponge (the note of 2026-11-02), before any T5 eval
bash $E/sim_task.sh 0 NovelSpongeOnPlate-v0 0 8 smoke_sim_sponge_iid98 --controller planner --planner mplib \
    --target-source sim --robot-mask joints --noise-seed 0 --layout-seed 98 --layout-per-episode
IID_EPISODES=8 TASK_SET=novel TASK_SOURCE=vp bash $E/planner_eval.sh 0 10320 smoke_vp_novel2_iid98 none iid98
# T5
IID_EPISODES=48 TASK_SET=novel bash $E/planner_eval.sh 0 10320 pseed31_novel_iid18_none none iid18
IID_EPISODES=48 TASK_SET=novel TASK_SOURCE=vp bash $E/planner_eval.sh 0 10320 vp_novel_iid18_none none iid18
IID_EPISODES=48 TASK_SET=novel bash $E/vla_eval.sh 0 10400 vla_novel_iid18 iid18 4
CKPT=$RUN_ROOT/vlapl/checkpoints/steps_${STEPS}_pytorch_model.pt IID_EPISODES=48 TASK_SET=novel \
    bash $E/vla_eval.sh 0 10400 vlapl_novel_iid18 iid18 4
# the instruction study's smoke tests (not part of the results): the planner reading the simulator's positions on the
# first 8 episodes of each choice scene on iid98
for t in ChoiceSpoonOnTowel-v0 ChoiceCarrotOnPlate-v0 ChoiceSpoonOnPlate-v0 ChoiceCarrotOnTowel-v0; do
  TASK_SET=choice bash $E/sim_task.sh 0 $t 0 8 smoke_sim_choice_iid98/$t --controller planner --planner mplib \
      --target-source sim --robot-mask joints --noise-seed 0 --layout-seed 98 --layout-per-episode
done
# the instruction study: the planner with the fixed rule and with the VLM's marks, VLA-0 and VLA-PL
IID_EPISODES=48 TASK_SET=choice bash $E/planner_eval.sh 0 10320 pseed31_choice_iid19_none none iid19
IID_EPISODES=48 TASK_SET=choice TASK_SOURCE=vp bash $E/planner_eval.sh 0 10320 vp_choice_iid19_none none iid19
IID_EPISODES=48 TASK_SET=choice bash $E/vla_eval.sh 0 10400 vla_choice_iid19 iid19 4
CKPT=$RUN_ROOT/vlapl/checkpoints/steps_${STEPS}_pytorch_model.pt IID_EPISODES=48 TASK_SET=choice \
    bash $E/vla_eval.sh 0 10400 vlapl_choice_iid19 iid19 4
```

The frontier-model study's plan arm (see "Frontier-model study" below). The frontier model's answers are made on a
machine of ours that holds the API key (`ANTHROPIC_API_KEY` in its environment, with the Anthropic SDK), never on
the GPU server: `FRAMES` is that machine's copy of the frames folders, and its answers files go to
`$RUNS_DIR/fanswers` on the server.

```bash
# Qwen3-VL's plan on the test sets
export TASK_SOURCE=vlm
bash $E/planner_eval.sh 0 10320 vlmplan_iid16_none none iid16
for set in synonyms attributes order reasoning; do
  INSTRUCTIONS=$set bash $E/planner_eval.sh 0 10320 vlmplan_words_${set}_paper_none none paper
done
IID_EPISODES=48 TASK_SET=harder bash $E/planner_eval.sh 0 10320 vlmplan_harder_iid17_none none iid17
IID_EPISODES=48 TASK_SET=novel bash $E/planner_eval.sh 0 10320 vlmplan_novel_iid18_none none iid18
unset TASK_SOURCE
# the first frames: the smoke test's (not part of the results: the first 8 episodes of each of the 17 tasks on
# iid98), then the test sets'
IID_EPISODES=8 bash $E/frames_eval.sh 0 frames_smoke_iid98 iid98
IID_EPISODES=8 TASK_SET=harder bash $E/frames_eval.sh 0 frames_smoke_harder_iid98 iid98
IID_EPISODES=8 TASK_SET=novel bash $E/frames_eval.sh 0 frames_smoke_novel_iid98 iid98
bash $E/frames_eval.sh 0 frames_iid16 iid16
for set in synonyms attributes order reasoning; do
  INSTRUCTIONS=$set bash $E/frames_eval.sh 0 frames_words_${set}_paper paper
done
IID_EPISODES=48 TASK_SET=harder bash $E/frames_eval.sh 0 frames_harder_iid17 iid17
IID_EPISODES=48 TASK_SET=novel bash $E/frames_eval.sh 0 frames_novel_iid18 iid18
# on the machine with the key: the frontier model's answers, one file for each frames folder (--dry-run counts
# the episodes without a request; a run goes on where the last one stopped and asks no episode twice)
for set in smoke_iid98 smoke_harder_iid98 smoke_novel_iid98 iid16 words_synonyms_paper words_attributes_paper \
    words_order_paper words_reasoning_paper harder_iid17 novel_iid18; do
  python -m examples.SimplerEnv.eval_files.planner.experiments.frontier_answers $FRAMES/frames_$set \
      fanswers/$set.jsonl
done
# the planner reading the saved answers: the smoke test, then the test sets (T1 also with harsh depth: the same
# answers, because the picture is the same)
export TASK_SOURCE=vlm
VLM_PLAN_ANSWERS=fanswers/smoke_iid98.jsonl IID_EPISODES=8 \
    bash $E/planner_eval.sh 0 10320 smoke_fplan_iid98_none none iid98
VLM_PLAN_ANSWERS=fanswers/smoke_harder_iid98.jsonl IID_EPISODES=8 TASK_SET=harder \
    bash $E/planner_eval.sh 0 10320 smoke_fplan_harder_iid98_none none iid98
VLM_PLAN_ANSWERS=fanswers/smoke_novel_iid98.jsonl IID_EPISODES=8 TASK_SET=novel \
    bash $E/planner_eval.sh 0 10320 smoke_fplan_novel_iid98_none none iid98
VLM_PLAN_ANSWERS=fanswers/iid16.jsonl bash $E/planner_eval.sh 0 10320 fplan_iid16_none none iid16
VLM_PLAN_ANSWERS=fanswers/iid16.jsonl bash $E/planner_eval.sh 0 10320 fplan_iid16_harsh harsh iid16
for set in synonyms attributes order reasoning; do
  VLM_PLAN_ANSWERS=fanswers/words_${set}_paper.jsonl INSTRUCTIONS=$set \
      bash $E/planner_eval.sh 0 10320 fplan_words_${set}_paper_none none paper
done
VLM_PLAN_ANSWERS=fanswers/harder_iid17.jsonl IID_EPISODES=48 TASK_SET=harder \
    bash $E/planner_eval.sh 0 10320 fplan_harder_iid17_none none iid17
VLM_PLAN_ANSWERS=fanswers/novel_iid18.jsonl IID_EPISODES=48 TASK_SET=novel \
    bash $E/planner_eval.sh 0 10320 fplan_novel_iid18_none none iid18
unset TASK_SOURCE
```

The frontier-model study's agent arm. The relay runs on the machine with the key and listens on that machine only;
the SSH tunnel that the same machine opens gives the GPU server its end, a Unix socket in a folder of ours there
(`SERVER_FOLDER`) that only its owner can open. `LIMIT` is the relay's guard in USD.

```bash
# a pipeline check with a made-up model (not part of the results; no request to any model): the tools, the steps
# and the records on the first 8 episodes of SimplerEnv's tasks on iid98
AGENT_MODEL=scripted IID_EPISODES=8 bash $E/agent_eval.sh 0 smoke_fagent_scripted_iid98 iid98
# on the machine with the key: the relay and the tunnel
python -m examples.SimplerEnv.eval_files.planner.experiments.agent_relay agent_relay.jsonl --limit-usd $LIMIT &
ssh -N -R $SERVER_FOLDER/agent_relay.sock:127.0.0.1:8765 $SERVER
# on the GPU server: the passes on iid98 that build the harness (PASS = 1, 2, 3; not part of the results)
export AGENT_RELAY=unix:$SERVER_FOLDER/agent_relay.sock
IID_EPISODES=8 bash $E/agent_eval.sh 0 smoke_fagent_pass${PASS}_iid98 iid98
IID_EPISODES=8 TASK_SET=harder bash $E/agent_eval.sh 0 smoke_fagent_pass${PASS}_harder_iid98 iid98
IID_EPISODES=8 TASK_SET=novel bash $E/agent_eval.sh 0 smoke_fagent_pass${PASS}_novel_iid98 iid98
# the test sets, whole sets only, in the order T5, T2, T3 (after the note that freezes the harness)
IID_EPISODES=48 TASK_SET=novel bash $E/agent_eval.sh 0 fagent_novel_iid18 iid18
for set in synonyms attributes order reasoning; do
  INSTRUCTIONS=$set bash $E/agent_eval.sh 0 fagent_words_${set}_paper paper
done
IID_EPISODES=48 TASK_SET=harder bash $E/agent_eval.sh 0 fagent_harder_iid17 iid17
unset AGENT_RELAY
```

The two-GPU VLA evals give each GPU half of the episodes as explicit chunks, e.g. `heldout_vla_g0` has
`TASK:0:8 TASK:8:16 TASK:16:24` for each task and `heldout_vla_g1` the episodes 24 to 47.

## Off-limits GPUs

Added on 2026-11-13.

**The problem.** A runner's `GPU` argument becomes `CUDA_VISIBLE_DEVICES`, which keeps CUDA on that GPU. It does not
keep the simulator's renderer there. The renderer is a Vulkan process, and the NVIDIA Vulkan driver opens a GPU when
the renderer starts (in `vkCreateInstance`), before any device is chosen: on our machine with two GPUs always the GPU
with the boot display, whichever GPU the eval runs on. Each simulator process then holds a graphics context of about
13 MiB there, with no load; the rendering and all computation are on the eval's GPU. The list of compute processes
(`nvidia-smi --query-compute-apps`) does not show such a context. `nvidia-smi pmon -c 1` does (type G), and so does
the GPU's used memory. SAPIEN's `device` option and the settings of Vulkan's device-select layer change which
device renders, not this context.

**The setting.** `OFF_LIMITS_GPUS=1` (nvidia-smi's indices, with commas). The runners of the table above then start
every simulator, every model server and the MPS daemon through `off_limits_gpus.py`, and they stop before anything
starts when the restriction does not work on the machine. Without the setting nothing changes. Any other command
can start through the launcher by hand, e.g. a script that is not one of these runners:

```bash
OFF_LIMITS_GPUS=1 python examples/SimplerEnv/eval_files/planner/experiments/off_limits_gpus.py COMMAND ARGUMENT...
OFF_LIMITS_GPUS=1 python examples/SimplerEnv/eval_files/planner/experiments/off_limits_gpus.py --check
```

**How.** The launcher asks the Linux kernel (Landlock, Linux 5.13 or later, no special rights) to forbid the command
and every process that it starts to open those GPUs' device nodes, and then runs the command. The driver does not
list a GPU whose node it cannot open, so Vulkan, CUDA and `nvidia-smi` in the command see the other GPUs only. It
takes access away from our own processes and from nothing else. Three details:

- A GPU's node is `/dev/nvidiaN` with N its device minor, which is not nvidia-smi's index (on our machine the two
  are swapped), so the launcher looks the nodes up at each start (`/proc/driver/nvidia/gpus/ADDRESS/information`).
- Landlock allows by list, so the launcher allows opening files in every entry of `/` except `/dev` and in every
  entry of `/dev` except those nodes. Nothing else is restricted. A file that is made later directly in `/` or in
  `/dev` cannot be opened by the command.
- The command starts only after a proof: each forbidden node was tried and could not be opened, and each other
  GPU's node could. Without Landlock, or with a GPU index that the machine does not have, it does not start.

**What was checked** (2026-11-13; two RTX A6000, driver 595, Linux 6.8 with Landlock version 4). Under the launcher,
Vulkan lists one NVIDIA device, the renderer and PyTorch use the eval's GPU, and `nvidia-smi` in the command lists
that GPU only. The first frames of 32 episodes (the first 8 of each of SimplerEnv's tasks on `iid98`, with
`frames_eval.sh`) are the same files, by their hashes, as without the launcher. While those simulators ran, the
other GPU's memory stayed at its value at rest in every sample (82), and `pmon` showed no process of the run on it.
Then the runners themselves with the setting (2026-11-14): `frames_eval.sh` wrote the same 32 files again, and a
short `vla_eval.sh` (4 episodes, 2 simulators, and the policy, SAM and VLM servers on the MPS daemon) ran to its end;
`pmon` showed no process of either run on the other GPU in 335 samples, one a second.

**Evals before this date.** Every simulator eval of this folder before 2026-11-13 ran without the setting, so each
of its simulator processes held that context on the machine's other GPU. No computation ran there, and the pictures
do not depend on it (the 32 first frames above).

**Limits.** Linux with Landlock only: with the setting, a runner stops on any other machine. Only the runners of the
table above read the setting; `auto_eval_scripts/run_eval.sh`, the Robocasa evals and the training scripts do not,
and need the launcher by hand. The tests (`tests/test_off_limits_gpus.py`) check the node lookup, the allow list,
the kernel's restriction on made-up files and the runners' calls; a GPU is needed for the checks of the paragraph
above.

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
`score_plan_check_v2 --dev --save plan_check_v2.json` (the dev step) and `score_plan_check_v2` (the test). The
VLM-plan study's: `score_vlm_plan` (all its tables and tests, with the role split and the other baselines),
`check_closed_loop closed_loop_vlmplan_iid15 --planner 'vlmplan_iid15_{}' --models 'vlmplan_model/asm_{}.json' --vla
vla31_iid15` (its closed-loop eval) and `score_costs vlmplan_iid15_harsh vla31_iid15 closed_loop_vlmplan_iid15
handoff_vlmplan_iid15` (its costs). The difficulty sweep's: `score_sweep`. The missing-depth study's:
`score_missing_depth`. The harder-task system study's: `score_harder_system`. The energy study's: `score_energy`.
The marriage study's: `score_marriage` (all its tables and tests). The VLM-router study's: `score_vlm_router`. The
instruction study's: `score_instruction`. The frontier-model study's: `score_frontier` (all its tables and tests,
for both arms). Any eval's costs (see "Costs" above):
`score_costs RUN...`.

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

A method. Its study, "VLM-plan study" below, was planned before any eval with it ran.

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
0.5 would reject them. The minimum is therefore not fixed here; the VLM-plan study (below) runs with the check
off and gives the ASM the SAM scores instead.

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

## VP-VLA's VLM marks

A method (since 2026-10-30): the planner reads the same visual prompts as VP-VLA instead of its fixed rule, so that
one language front-end serves both. Its study is planned before any eval with it runs.

VP-VLA's VLM (`vlm_server_subtask_simpler.py`) splits the instruction into subtasks (when it gives none, VP-VLA
splits the instruction at "and" and commas). For the current subtask it names the object to move and, in a placing
subtask, the target; VP-VLA's SAM turns these names into its crosshair (the object) and its box (the target). With
`TASK_SOURCE=vp` (`--task-source vp`) the planner asks the same VLM server the same questions on the first frame,
one for each subtask. VP-VLA asks about the current subtask only, at the first frame and at each gripper change; the
planner plans from its first look, so it asks about every subtask there. Its question about the first subtask is the
one VP-VLA asks at the first frame, so with the VLA's VLM settings (`--seed-per-query`) both get the same answer.
Each subtask that names a target ends a pick-and-place move of the object that it names (else of the last object
named before it). Of two moves of the same object in a row only the later one is kept, because a pick subtask that
also names a target only looks ahead to its place subtask. SAM then finds each name as with the VLM plan, and the
planner does the moves one after another. When no subtask names a target, a planner-only episode ends, and in the
selector mode the VLA gets the episode before the ASM decides. Unlike VP-VLA, the planner never falls back to the
fixed rule's object when the VLM names none.

Why every subtask on the first frame (exploratory, from VP-VLA's eval logs of 2026-10-30): the VLM named a target in
only 27 of 743 first answers, which are about the pick subtask, and in 472 of 695 second answers, after the grasp.
Asking about the current subtask only would leave the planner without a target at its first look, where the ASM's
features need one.

The records add `task_source` (`vp`), `vp_subtasks`, `vp_subtasks_from` (`vlm`, or `rule` for VP-VLA's split),
`vp_marks` (for each subtask: the object and target that the VLM names, and its answer), `vlm_plan_feasible` (a
move was found), `steps_done` and `vlm_plan_sam` (as with the VLM plan).

## VLM-plan study

The plan below was fixed on 2026-10-23, before any eval with the VLM plan ran, and `score_vlm_plan` follows it. The
results do not change it. A result that is not in the plan is reported as exploratory.

**Question.** A common alternative to our architecture puts a language model or a VLM in front of a motion planner
(VLM -> planner). With the VLM plan (see "VLM plan" above), VLM -> planner is our planner branch alone, and our
system is the VLM plan, then the ASM, then the planner or the VLA. Does adding the ASM and the VLA to VLM -> planner
raise the success rate? A VLM changes how the instruction is read, not what the depth camera measures: in the
two-model study, with instructions that the fixed rule reads correctly, the planner fell from 90.0% with clean depth
to 49.0% with harsh depth.

**The system.** On the first frame, the VLM plan. A plan that is not pick-and-place gives the episode to the VLA
(`selector.vlm_plan_gate`). The two-model ASM decides the other episodes from the planner's first look, which uses the
names of the VLM's first move: the planner when P(planner succeeds) >= P(VLA succeeds), else the VLA. This is the
selector mode with `TASK_SOURCE=vlm`.

**Tasks and layouts.** SimplerEnv's 4 tasks with their own instructions. Training set: `iid11` (the two-model
study's). Test set: `iid15` (layout seed 15, a random layout of its own for each episode, 100 per task), new for this
study: no episode of it ran before this plan. (`iid12`, the two-model study's test set, was seen.)

**Evals.** All with SAM 3.1 (the default), noise seed 0 and planner seed 0; the VLA with the fast settings and VLM
seed offset 0.

- Planner-only with the VLM plan (`TASK_SOURCE=vlm`): `vlmplan_iid11_{none,harsh}` and `vlmplan_iid15_{none,harsh}`.
- VLA-only: `vla31_iid11` (also the learned-features data) and `vla31_iid15`.
- Planner-only with the fixed rule, for the secondary results: `pseed31_iid11_harsh` (also the learned-features data)
  and `pseed31_iid15_harsh`.
- The closed-loop eval `closed_loop_vlmplan_iid15` (harsh depth, `TASK_SOURCE=vlm`) with the saved ASMs
  `vlmplan_model/asm_{level}.json`.

The ASMs are trained and saved before any eval of the test set runs. Each eval runs alone on GPU 0, so that its GPU
energy is its own (see "Costs"). About 6 GPU hours, besides the learned-features data. The commands are above.

**SAM score check: off** (`VLM_PLAN_MIN_SCORE=0`) in every eval. "VLM plan" above first said that this plan would fit
a minimum score on the training set. Instead, the first move's SAM scores are candidate features of P(planner)
(below), so the training set decides through the ASM whether a low score lowers the planner's chance; a fixed minimum
would take that choice away.

**Models.** One ASM per depth level (clean and harsh), trained on `iid11` only, with `selector.py --rule two-model`
as in the two-model study: each success model is a logistic regression (L2 penalty 1, standardized inputs) on its
features and one column per task. P(VLA succeeds) uses `grasp_turn_rad` and `grasp_reach_m`. P(planner succeeds) uses
one of these feature lists, the candidates fixed in "VLM plan" above:

1. `plan_ok`, `plan_path_ratio` (the two-model study's);
2. `plan_ok`, `plan_path_ratio`, `grasp_sam3_score`, `place_sam3_score` (the best SAM score of the first move's
   object and target names on the first frame).

The list with the lower log-loss in 10-fold cross-validation on the training set is used (`--planner-candidates`,
`selector.choose_features`; the first list on a tie). The folds hold out whole episode indices (random generator
seed 0), because the 4 tasks of an episode index share their layout. The episodes that the gate gives to the VLA are
not training data.

**Decision rule.** The gate, then the planner when P(planner) >= P(VLA), else the VLA.

**Planned tests.** On the 400 episodes of `iid15` with harsh depth (with harsh depth the planner fails most, so the
choice matters most), the system above against, in this order:

1. VLM -> planner alone (`vlmplan_iid15_harsh`);
2. the VLA alone (`vla31_iid15`).

The system is scored offline, as in the earlier studies: each episode's choice from the planner-only eval's first
look, and the outcome of the chosen method's eval of that episode. Each test is the exact two-sided paired sign-flip
test at 0.05, with one value per episode index: its sum over the 4 tasks of the difference in success (the 4 tasks
of an episode index share their layout, so they are not independent). A test is a win when it is significant with
the system higher, and a test counts only when every test before it was a win. Only the episode indices with all 4
tasks in every eval count. Intervals: 95% percentile bootstrap of the mean difference; each draw takes the episode
indices again, with replacement (10,000 draws).

**Secondary (reported, no claims).** The clean depth level; each method per task and the oracle; the gate alone
(VLM -> planner, and the VLA when the plan is not pick-and-place); what the VLM planned (how often it found
pick-and-place moves, its names and their SAM scores); each candidate's cross-validated log-loss, and the success
models' AUC, within-task AUC, log-loss, Brier score and calibration on the test set; the VLM plan against the fixed
rule on the same episodes, planner-only and with the ASM (the fixed rule's ASM has the two-model study's features and
is trained on `pseed31_iid11_harsh` and `vla31_iid11`); the closed-loop eval against the offline result
(`check_closed_loop`); each eval's cost per episode (`score_costs`: seconds, model requests and GPU energy), where the
closed-loop eval's cost is the whole system's.

**Change of 2026-10-25, before any of the study's evals ran.** Dr. Zhu asked how the system compares with other
ways to combine a planner and a VLA, and what makes it better. Tests 1 and 2 above stay as they are, and these are
added (`score_vlm_plan` does them):

- **Two more planned tests**, on the same 400 episodes and with the same test, which count only when tests 1 and 2
  were both wins. The system against:

  3. the role split: VLM -> planner moves the hand to above the object, and then the VLA does the rest of the
     episode. Hybrid systems often divide the work this way (a planner for the free-space motion, a learned policy
     for the contact). A plan that is not pick-and-place, or a planner that fails before the hand-off, gives the
     whole episode to the VLA. It is a closed-loop eval of its own, `handoff_vlmplan_iid15`, with harsh and clean
     depth (`selector_eval.sh` with `CONTROLLER=planner_then_vla`; about 4 GPU hours on GPU 0);
  4. the per-task choice: the gate, then for each task the method with the higher success rate on the training
     set's episodes that the gate does not give to the VLA (the planner on a tie). It is a router that chooses per
     task and does not look at the episode.

  Tests 3 and 4 are corrected for two tests by Holm's method at 0.05: the smaller p must be below 0.025, and then
  the other below 0.05. A test is a win when it is significant with the system higher. If the role-split eval did
  not run, test 3 has no result and counts as p = 1. The role split's value for an episode index is its own
  outcome, summed over the 4 tasks as for the other setups; an index that its eval did not run in full is left out
  of test 3 only.
- **Secondary baselines** (the same test, no claims), each behind the same gate and trained on the same episodes as
  the ASM: rule A, the planner when P(planner) >= 0.5 (the ASM's planner model alone); rule C, the planner when
  P(planner) >= the task's VLA success rate on the training set; a retrieval router, the method with more successes
  on the k most similar training episodes of the same task (Euclidean distance on the ASM's features, each
  standardized on the training set; the planner on a tie), with k from 1, 3, 5, 9, 15, 25 and 45 chosen by the
  routed success in 10-fold cross-validation on the training set (folds by episode index, seed 0; the larger k on
  a tie). Also a random choice with the system's share of planner picks (its expected success, no test), and the
  share of the oracle's gain over VLM -> planner that the system gets.
- **What makes the system better** (`mechanism.py`; reported at each depth level, with no tests; the failure
  dependence and the system minus the oracle have bootstrap intervals, and the learning curve the spread of its
  draws):
  - complementarity: how many test episodes both methods solve, only VLM -> planner, only the VLA, or neither. The
    oracle's gain over the better method equals the share of episodes that only the worse method solves, so a
    choice per episode can gain only on those;
  - failure dependence: within each task, P(the VLA succeeds | VLM -> planner failed) - P(the VLA succeeds |
    VLM -> planner succeeded) (Mantel-Haenszel), with a bootstrap interval over the episode indices. Near 0, the
    two methods fail on different episodes; below 0, on the same ones;
  - the gain decomposition: of the episodes that only one method solves, how many the system gave to that method;
  - the success and the gain over the VLA that the ASM's own probabilities predict, against the real ones;
  - ablations: the ASM trained again without each of its features, and with only the task columns in one of its
    success models;
  - a learning curve: the ASM trained on 10%, 25% and 50% of the training set's episode indices (20 random draws
    each, seed 0; a draw without any episode of a task is skipped, and the table gives the draws used) and on all
    of them;
  - the system minus the oracle.

## Difficulty sweep

The plan below was fixed on 2026-10-25, before any of the study's evals ran, and `score_sweep` follows it. The
results do not change it. A result that is not in the plan is reported as exploratory.

**Question.** A choice per episode can only gain on the episodes that one method solves and the other does not. In
an exploratory look at the two-model study's test set (2026-10-25), harsh depth gave many such episodes (89 that
only the planner solved and 108 that only the VLA solved, of 400) and clean depth few. As the depth noise grows, the
planner falls and the better of the two methods changes from the planner to the VLA. Does the system stay above the
better method across the whole range, from clean depth to twice the harsh level?

**Levels** (`geometry.DepthNoise`). Seven: clean (`none`), and harsh's errors times 0.25, 0.5, 0.75, 1 (`harsh`),
1.5 and 2 (`harsh_x0.25`, `harsh_x0.5`, `harsh_x0.75`, `harsh_x1.5`, `harsh_x2`). The scale multiplies the depth
noise of each pixel, the chance that a pixel or an edge pixel has no depth (at most 1) and the camera pose error;
the threshold that says what counts as a depth edge stays. All levels draw the same random numbers, so in an
episode's first picture a level's errors are harsh's times its scale, and its pixels without depth include those
of every smaller scale. The camera picture is clean at every level.

**System.** The VLM-plan study's (the VLM plan, its gate, then a two-model ASM), with one ASM for every level. It
is not told the level: it is trained on the training set's clean and harsh evals together (`vlmplan_iid11_none`
and `vlmplan_iid11_harsh` with `vla31_iid11`; `selector.train_pooled`), and P(planner)'s features are chosen from
the VLM-plan study's two candidates by the same cross-validation, with folds that hold out an episode index at both
levels at once. The VLA does not use depth, so one VLA-only eval serves every level.

**Evals.** The VLM-plan study's test set `iid15`, planner-only with the VLM plan at the 5 new levels:
`vlmplan_iid15_harsh_x{0.25,0.5,0.75,1.5,2}` (2,000 episodes; SAM 3.1, noise seed 0, planner seed 0). The clean
and harsh levels (`vlmplan_iid15_{none,harsh}`) and the VLA-only eval (`vla31_iid15`) are the VLM-plan study's.
About 2 GPU hours on GPU 0. The commands are above.

**Planned test.** The system against the better of VLM -> planner and the VLA at each level (the one with the
higher success rate on the test set at that level, which favors this baseline). The exact two-sided paired
sign-flip test at 0.05, with one value per episode index: its sum, over the 7 levels and the 4 tasks, of the
difference in success. It is a win when it is significant with the system higher. Only the episode indices with all
4 tasks at every level count. Interval: 95% percentile bootstrap of the mean difference; each draw takes the
episode indices again, with replacement (10,000 draws).

**Secondary (reported, no claims).** At each level: each method, the better one, the system and the oracle, the
oracle's gain over the better method and the share of it that the system gets, and how many episodes the system
gives the VLA; the system against VLM -> planner and against the VLA (the same test); the complementarity, the
failure dependence and the gain decomposition (see "VLM-plan study"). At the clean and harsh levels: the sweep's
ASM against the VLM-plan study's ASM of that level (trained on that level only).

## Missing-depth study

The plan below was fixed on 2026-10-25, before any of the study's evals ran, and `score_missing_depth` follows it.
The results do not change it. A result that is not in the plan is reported as exploratory.

**Question.** Some surfaces give a depth camera no depth at all: glass, mirror-like metal, or black cloth that takes
in the camera's light. The color picture still shows them. When the place target is such a surface, the planner
cannot place the object on it, but the VLA, which does not use depth, can. Can the system's first look find these
episodes and give them to the VLA?

**The change to the scene** (`MISSING_DEPTH=target`, `geometry.MissingDepth`). In each episode, with chance 1/2,
no pixel of the place target has depth (the towel, the plate or the yellow cube). The eggplant task keeps its depth,
because its basket is part of the sink, not an object of its own. The draw depends only on the noise seed and the
episode, in a random stream of its own, so the other noise stays the same. The rest of the depth is clean, and the
camera picture does not change. The records log `target_depth_missing`.

**What the planner does** (from its code, before any eval ran). The first look finds the place target only for the
ASM's features, and the planner does not stop when the target has no depth. It finds a path to above the object
(`plan_ok` is 1), picks the object up and fails after the lift, when it looks for the target again ("place target
not found"). In the first look only the target's depth features show it: the share of its mask's pixels with depth
(`place_valid_depth_frac`) and its number of 3D points (`place_points`) are 0, and the features of its shape are not
computed. Of the ASM's candidate features, only `place_valid_depth_frac` shows it.

**System.** The VLM-plan study's: the VLM plan, its gate, then a two-model ASM trained on the training set's
missing-depth eval. P(planner)'s features are chosen by the VLM-plan study's cross-validation from three candidates:
the VLM-plan study's two, and the two-model study's features with the share of each mask's pixels that have depth
(`plan_ok`, `plan_path_ratio`, `grasp_valid_depth_frac`, `place_valid_depth_frac`). The first wins a tie.

**Tasks and evals.** SimplerEnv's 4 tasks with their own instructions, on the VLM-plan study's training set
(`iid11`) and test set (`iid15`). Planner-only with the VLM plan and missing target depth: `miss_iid11_none` and
`miss_iid15_none` (800 episodes; SAM 3.1, noise seed 0, planner seed 0). VLA-only: `vla31_iid11` and `vla31_iid15`
(the VLM-plan study's: the VLA does not use depth). Under 1 GPU hour on GPU 0. The commands are above.

**Planned tests.** On the 400 episodes of `miss_iid15_none`, the system against, in this order:

1. VLM -> planner alone (the same planner-only eval);
2. the VLA alone (`vla31_iid15`).

The tests, their order and their intervals are those of the VLM-plan study's tests 1 and 2.

**Secondary (reported, no claims).** Each setup per task, the gate alone and the oracle. The system against (the
same test): the planner first (the gate, then the planner whenever its first look found a path, else the VLA); the
VLM-plan study's clean ASM (trained on `vlmplan_iid11_none`, which had no missing depth); the per-task choice. A
random choice with the system's share of planner picks (expected). VLM -> planner, the VLA, the system, the planner
first, the clean ASM and the oracle on the episodes with and without the target's depth, and why VLM -> planner
failed in each. VLM -> planner against the same episodes with the target's depth (`vlmplan_iid15_none`). The candidates' cross-validated log-loss, and the success models' AUC, within-task
AUC, log-loss, Brier score and calibration. What makes the system better (see "VLM-plan study").

## Harder-task system study

The plan below was fixed on 2026-10-25, before any of the study's evals ran, and `score_harder_system` follows it.
The results do not change it. A result that is not in the plan is reported as exploratory.

**Question.** The VLM-plan study tests the system on SimplerEnv's own tasks and instructions. With reworded
instructions and harder scenes, where the planner's fixed rule fails, does the whole system (the VLM plan, its gate
and the ASM) still beat VLM -> planner, the VLA, and a choice per task?

**Tasks.** Harsh depth throughout, and the VLM plan. SimplerEnv's 4 tasks with their own instructions and with each
of the 4 reworded sets (`instruction_sets.py`, see "Reworded-instruction study"), and the 5 harder scenes
(`harder_scenes.py`, see "Harder-scene study"): 9 tasks, as a reworded instruction keeps its task.

**Training set.** The VLM-plan study's training eval with harsh depth (`vlmplan_iid11_harsh` with `vla31_iid11`:
SimplerEnv's tasks, 400 episodes) and the harder scenes' fixed layouts (`vlmplan_harder_paper_harsh` with
`vla_harder_paper`: 352 episodes). No reworded instruction is in it. One ASM with a task column for each of the 9
tasks is trained on both together (`selector.train_pooled`), with P(planner)'s features chosen from the VLM-plan
study's two candidates by the same cross-validation. The folds hold out an episode index of each layout set; in the
harder scenes' fixed layouts the two episodes of an xy config differ only in the turns, so they are held out
together.

**Test set.** The `iid13` layouts (layout seed 13; see "VLM plan-check v2 study"): SimplerEnv's 4 tasks, 24
episodes each, with their own instructions and with each reworded set (5 x 96 episodes; episode N has the same
layout in every set), and the 5 harder scenes, 48 episodes each (240 episodes): 720 episodes.

**Evals.** Planner-only with the VLM plan and harsh depth (SAM 3.1, noise seed 0, planner seed 0):
`vlmplan_harder_paper_harsh` (training), and `vlmplan_own_iid13_harsh`, `vlmplan_words_{set}_iid13_harsh` and
`vlmplan_harder_iid13_harsh` (test). VLA-only: `vla_harder_paper` (the harder-scene study's) and
`vla_{own,words_SET,harder}_iid13` (the plan-check v2 study's; they run for this study whatever that study's dev
step says). For the secondary results, planner-only with the fixed rule and harsh depth:
`pseed31_{own,words_SET,harder}_iid13_harsh`. About 1.5 GPU hours on GPU 0 for these planner-only evals, besides
the VLA-only evals of the other studies. The commands are above.

**Planned tests.** On the 720 test episodes, the system against, in this order:

1. VLM -> planner alone;
2. the VLA alone;
3. the per-task choice (as in the VLM-plan study: the gate, then each task's better method on the training set).

Each is the exact two-sided paired sign-flip test at 0.05, and each counts only when every test before it was a win
(significant with the system higher). In `iid13` an episode index is one layout in every instruction set, and the
tasks of a scene share it, so each test takes one value per episode index: its sum over every instruction group and
task (indices 24 to 47 have the harder scenes only). The tests need the planner-only and the VLA-only evals of every
instruction group. Only the design's episodes count (0 to 23 of SimplerEnv's tasks in each instruction group, 0 to 47
of the harder scenes), and of these only the episode indices that have every task of every group in both evals.
Intervals: 95% percentile bootstrap of the mean difference per episode; each draw takes the episode indices again,
with replacement (10,000 draws).

**Secondary (reported, no claims).** Each setup per instruction group, with the gate alone and the oracle;
VLM -> planner, the VLA, the system and the oracle per task; rule A, rule C and the retrieval router (see "VLM-plan study"; the same test); a random choice with the system's
share of planner picks (expected); the success models' AUC, within-task AUC, log-loss, Brier score and calibration;
what makes the system better (see "VLM-plan study"); what the VLM planned in each task. The VLM plan against the
fixed rule on the same episodes: planner-only in every group, and the fixed rule's ASM (the two-model study's
features, trained on `pseed31_iid11_harsh` and `vla31_iid11`, so on SimplerEnv's tasks only).

## Energy study

The plan below was fixed on 2026-10-30, before any energy run, and `score_energy` follows it. The results do not
change it. A result that is not in the plan is reported as exploratory.

**Question.** The VLA runs its models on the GPU at every step of an episode. The planner uses the GPU only for SAM
(to find the object and the target) and plans on the CPU. So every episode that the ASM gives to the planner should
cost much less energy than the same episode with the VLA. How much GPU energy per episode does the ASM system use,
against the VLA alone and the planner alone?

**Systems.** On the same episodes, all with SAM 3.1:

- the VLA alone (VP-VLA with the fast settings);
- the planner alone (the fixed rule, planner seed 0, noise seed 0);
- the ASM system: the two-model ASM in the closed-loop eval (the planner when P(planner succeeds) >= P(VLA
  succeeds), else the VLA). An episode that goes to the VLA also pays for the planner's first look, which the ASM
  needs in every episode.

**ASMs.** One per depth level (clean and harsh), trained as in the two-model study (`selector.py --rule two-model`,
the same features) on `iid11` with SAM 3.1: `pseed31_iid11_{none,harsh}` and `vla31_iid11`. They are saved as
`energy_model/asm_{level}.json` before any energy run. (The two-model study's ASMs were trained with SAM 3, and a
saved ASM runs only with the SAM model it was trained on.)

**Episodes.** The first 16 episodes of each of SimplerEnv's 4 tasks in `iid15`, with the tasks' own instructions:
64 episodes per eval. The planner and the ASM system run with harsh depth and with clean depth. The VLA does not read
depth, so one VLA eval serves both levels.

**Measurement.** Each energy run (`ENERGY=1`) runs one sim at a time, alone on GPU 0 with no other user's job. Each
episode records the GPU's energy from the model's reset to the episode's last step, read from the GPU's own energy
counter (NVML's total energy, `energy.py`). The counter advances about every 0.1 s, so it is read once per episode,
not per request. Before its episodes, each run measures:

1. the GPU's idle power with the run's servers loaded: 60 s, after 30 s of settling (`idle_power.json`);
2. the same episodes with a controller that does nothing (`--controller null`: the hand stays still and the episode
   runs to its last step), in `RUN_null`. This is the simulation's own GPU energy per step on each task, with the
   same servers loaded.

An episode's **controller energy** is its GPU energy minus its steps times its task's simulation energy per step. It
is what the controller adds to the simulation: its models' GPU work, and the GPU's idle power while it waits for the
CPU. This is the main measure, because a real robot has no simulator to pay for.

**Planned test.** With harsh depth, the ASM system's closed-loop eval against the VLA alone, controller energy per
episode: the exact two-sided paired sign-flip test at 0.05 (all sign patterns), with one value per episode index (the
sum over the 4 tasks of the ASM system's energy minus the VLA's), because the 4 tasks of an index share their layout.
Only the episode indices with all 4 tasks in both evals count. A win: significant, with the ASM system lower.
Interval: 95% percentile bootstrap of the mean difference per episode; each draw takes the episode indices again,
with replacement (10,000 draws). Harsh depth is the planned level because it is the conservative case: with harsh
depth the planner fails most, so the ASM gives the most episodes to the VLA.

**Secondary (reported, no claims).** For each depth level and each system: the mean controller energy per episode
(with its interval), the saving against the VLA alone, the energy per successful episode (with its interval), and
kWh per million episodes (the mean in other units, not a forecast). The ASM system offline (each episode's
closed-loop choice, with the planner-only and VLA-only evals' energies) as a check of the closed-loop eval. The
oracle after the fact (the planner when it succeeds or when the VLA fails too, else the first look and the VLA). The
same test with clean depth. The total GPU energy (with the simulation), the energy above idle (controller energy
minus the idle power times the controller's seconds), a CPU estimate, and the wall-clock seconds. Each run's idle
power and simulation energy per step. The success rates of these evals: they are small (64 episodes per eval), so
the success claims stay with the two-model and VLM-plan studies.

**CPU estimate.** The server's CPU (an AMD Threadripper PRO 7975WX: 32 cores, 350 W TDP) has an energy counter
(RAPL) that only root can read. So the CPU energy is an estimate: the controller's CPU seconds times 350 W / 32. It
matters most for the planner, which plans on the CPU.

**Limits.** GPU energy and the CPU estimate only: no motors, camera or network. One GPU model (an RTX A6000), the
VLA's fast settings and one request at a time; a server that batches requests, or another GPU, changes the joules,
so the main result is the ratio between the systems. The simulation's share assumes that its energy per step is the
same with and without a controller; the GPU may run at another clock speed when the simulation runs alone, so the
total GPU energy is reported too.

**Evals.** About 3.5 GPU hours on GPU 0, and about 10 minutes for `pseed31_iid11_none`. The commands are above.

## Marriage study

The plan below was fixed on 2026-10-30, before any of the study's demonstrations, training runs or evals, and
`score_marriage` follows it. The results do not change it. A result that is not in the plan is reported as
exploratory.

**Question.** The studies above choose the planner or the VLA for each episode, so they can never beat the oracle
of the two. Here each one trains the other: the planner's successful episodes, also with reworded instructions,
become training data for VP-VLA, and VP-VLA's VLM tells the planner what to move (`TASK_SOURCE=vp`, see "VP-VLA's
VLM marks"). Does the VLA learn from the planner (H1, H2), does it then beat every option that needs no training on
instructions it never saw (H3), and does the married system beat the current one (H4)? Does choosing where to
collect help (H5)?

**Tasks.** Training uses SimplerEnv's 4 tasks with their own instructions only. The harder scenes are in no
training set: the fixed rule, which collects the demonstrations, cannot read their two-plate and two-step
instructions, and the visual-prompt labels take one goal. So T3 tests scenes that no model was trained on.

**Layout sets.** None of them ran before this plan.

| Set | Layouts | Use |
|---|---|---|
| `iid100` | layout seed 100, a random layout for each episode | the demonstrations' pool (training) |
| `iid101` | layout seed 101 | the planner's corrections (training) |
| `iid16` | layout seed 16, 100 per task | T1: SimplerEnv's 4 tasks with their own instructions (400 episodes) |
| `paper` | SimplerEnv's 24 layouts per task | T2: the 4 reworded sets of `instruction_sets.py` (384 episodes) |
| `iid17` | layout seed 17, 48 per task | T3: the 5 harder scenes of `harder_scenes.py` (240 episodes) |

T4 is T1 with the planner at the 3 depth levels (clean, realistic and harsh); the VLA does not read depth. The ASMs
of H4 are trained on `iid11` (the two-model study's training set) and on the harder scenes' fixed layouts (`paper`),
as in the harder-task system study; the trained VLAs never saw these layouts. No training rewording is a test
instruction or uses a test family's concept words: their exclusion list was fixed before any rewording was written
(see "Demonstrations" under "Evals (GPU)").

**Demonstrations.** All with SAM 3.1, clean depth and camera, noise seed 0 and planner seed 0, recorded with
`RECORD_DEMOS=1` and converted by `planner_demos_to_lerobot.py --visual-prompts` (the visual prompts from the
simulator's masks); `--reword` adds a training rewording of each of the four families to every episode, so each
episode is written 5 times. Each model gets 250 demonstrations of each task (1,000 in all):

- **VLA-P**: the planner (fixed rule) on the pool's episodes 0 to 699 of each task (`demos_planner_iid100`, 2,800
  episodes); the first 250 successful episodes of each task, in episode order, so at random layouts.
- **VLA-PL**: the same 1,000 episodes, with `--reword`.
- **VLA-S** (the control): VLA-0 on the same pool (`demos_vla_iid100`; fast settings, VLM seed offset 0), episodes
  0 to 399 of each task and then 100 more at a time for a task until it has 250 successes; the first 250 successful
  episodes of each task, in episode order, with `--reword`. VLA-S has VLA-PL's numbers and rewordings, from the same
  pool of layouts, with the VLA's own actions instead of the planner's.
- **VLA-PLT**: 250 successful episodes of each task from the same planner pool, chosen from each episode's first
  look. The score is P(planner succeeds) x (1 - P(VLA succeeds)) of the energy study's clean-depth ASM
  (`energy_model/asm_none.json`, trained on `pseed31_iid11_none` and `vla31_iid11`). Of each task, 188 are the
  successful episodes with the highest scores (the lower episode index on a tie), and 62 are drawn at random
  (seed 0) from the task's other successful episodes, for coverage. With `--reword`.
- **VLA-PLTD**: VLA-PLT's set and one round of the planner's corrections (see "Corrections" above): VLA-PLT on
  `iid101` with `CONTROLLER=vla_then_planner TAKEOVER_STEPS="10 40"`, clean depth (`demos_corrections_iid101`),
  episodes 0 to 99 of each task and then 100 more at a time until a task has 250 successful corrections (the planner
  took over, holding or to grasp, and the episode succeeded); the first 250 of each task, with `--reword`.

**Models.** VLA-0 is the released VP-VLA. The other 5 are fine-tuned from it with `run_finetune_1gpu.sh`, each on its
own set (VLA-PLTD's mixture has both of its sets), all with the same settings and the same number of steps. The
settings (the steps, the batch, the learning rates and the frozen layers) are fixed after the smoke tests, in a dated
note in this section, before the first model of the study is trained.

**Smoke tests first** (not part of the results). The pool's first 20 episodes of each task, recorded and
converted: the clip report, and the eggplant's box on rendered frames. If more than 1% of any action number lies
outside the checkpoint's range (training clips it there), the planner's steps in demonstrations get a limit that
keeps them inside the range, fixed in a dated note before the demonstrations run. Then a short training run on these
demonstrations measures the memory and the speed. No smoke test evaluates a model.

**Baselines without training** (for H3). **VLA-0 reading the subtask**: VLA-0 whose action head gets the current
subtask that VP-VLA's VLM writes (the text that the visual prompts come from) instead of the raw instruction, or the
raw instruction when the VLM gives none. **The planner reading the VLM's marks**: `TASK_SOURCE=vp`, clean depth.

**Evals.** All with SAM 3.1, noise seed 0 and planner seed 0; the VLAs with the fast settings and VLM seed offset 0.
The VLA-only evals, where `M` is a trained model's prefix (`vlap`, `vlapl`, `vlaplt`, `vlapltd`, `vlas`) and `S` a
reworded set:

| | T1 | T2 | T3 | the ASMs' training sets |
|---|---|---|---|---|
| VLA-0 | `vla31_iid16` | `vla_words_S_paper` (the reworded-instruction study's) | `vla_harder_iid17` | `vla31_iid11`, `vla_harder_paper` |
| each model `M` | `M_iid16` | `M_words_S_paper` | VLA-PL: `vlapl_harder_iid17` | VLA-PL: `vlapl_iid11`, `vlapl_harder_paper` |
| VLA-0 reading the subtask | | `vlasub_words_S_paper` | | |

The planner-only evals with the fixed rule: `pseed31_iid16_{none,realistic,harsh}` (T1 and T4), `words_S_paper`
(T2: the reworded-instruction study's, clean depth), `pseed31_harder_iid17_none` (T3), and `pseed31_iid11_{none,harsh}`
and `pseed31_harder_paper_none` (the ASMs' training sets). With the VLM's marks (`TASK_SOURCE=vp`):
`vp_iid16_{none,realistic,harsh}`, `vp_words_S_paper_none`, `vp_harder_iid17_none`, `vp_iid11_{none,harsh}` and
`vp_harder_paper_none`. The commands are above.

**Planned tests.** In this fixed order, each the exact two-sided paired sign-flip test at 0.05, and each counting only
when every test before it was a win (significant, with the first one higher):

| # | Question | Comparison | Test sets |
|---|---|---|---|
| H1 | Does the planner teach the VLA? | VLA-P vs. VLA-0 | T1 (400 episodes) |
| H2 | Is it the planner's data, not just more data from the simulator? | VLA-PL vs. VLA-S | T1 and T2 (784) |
| H3 | Does the married VLA beat every option that needs no training on new language? | VLA-PL vs. each of the planner (fixed rule), VLA-0, VLA-0 reading the subtask and the planner reading the VLM's marks | T2 (384) |
| H4 | Does the married system beat the current system? | the married system vs. the current system (below) | T1 with harsh depth, T2 and T3 (1,024) |
| H5 | Does choosing where to collect help? | VLA-PLT vs. VLA-PL | T1 and T2 (784) |

H3 is an intersection-union test: it is a win when VLA-PL is significantly higher than each of the four, each by the
same test at 0.05, so it needs no correction; its p is the largest of the four.

Each test takes one value per unit, the sum of its episodes' differences in success, because episodes that share a
layout are not independent: an episode index of T1 (its 4 tasks share the layout), a layout of T2 (one task's
episode in the 4 reworded sets) or an episode index of T3 (its 5 scenes share it). Only the design's episodes count
(T1: 0 to 99 of each task; T2: 0 to 23 of each task in each set; T3: 0 to 47 of each scene), and of these only the
units that every eval of the test has in full. Intervals: 95% percentile bootstrap of the mean difference per
episode; each draw takes the units again, with replacement (10,000 draws).

**The systems of H4.** Both are scored offline, as in the studies above: each episode's choice from the first look
of its planner-only eval, and the outcome of the chosen method's eval of that episode.

- The current system: the planner with its fixed rule, VLA-0 and the two-model ASM, with the two-model study's
  features (P(planner) from `plan_ok` and `plan_path_ratio`, P(VLA) from `grasp_turn_rad` and `grasp_reach_m`, and
  one column per task in each).
- The married system: the planner reading the VLM's marks, VLA-PL, and the same ASM trained on their outcomes. An
  episode where the VLM marks no target goes to the VLA first (`selector.vlm_plan_gate`), and such episodes are not
  training data. VLA-PL is fixed here, so that no test result picks the married system's VLA.

Each system has one ASM per depth level. Harsh (T1): trained on `iid11` with harsh depth. Clean (T2 and T3): trained
on `iid11` with clean depth and on the harder scenes' fixed layouts together, with a column for each of the 9 tasks
(as `selector.train_pooled`); a reworded instruction keeps its task's column. The current system's are trained on
`pseed31_iid11_{harsh,none}` with `vla31_iid11` and on `pseed31_harder_paper_none` with `vla_harder_paper`; the
married system's on `vp_iid11_{harsh,none}` with `vlapl_iid11` and on `vp_harder_paper_none` with
`vlapl_harder_paper`.

**Secondary (reported, no claims).**

- Every eval's success rate on T1, on each reworded set of T2 and on T3; each model against VLA-0 on each (exact
  McNemar).
- VLA-PLTD vs. VLA-PLT on T1 and T2. This comparison and those below use the planned tests' test, except where
  McNemar is named.
- The student against its teacher (T4): VLA-PL and VLA-0 vs. the planner with its fixed rule on T1, at each depth
  level.
- The planner reading the VLM's marks vs. its fixed rule, planner-only: on T1 at each depth level, on T2 and on T3.
- H4 on each test set: each system's planner-only, VLA-only, ASM, per-task choice (for each task the method with the
  higher success rate on the ASM's training episodes; the planner on a tie) and oracle; how many episodes both
  methods solve, only one, or neither; the married system against the old pair's oracle (after the fact, the better
  of the fixed rule and VLA-0 in each episode), on each test set and each reworded set; the success models' AUC,
  within-task AUC, log-loss, Brier score and calibration.
- Of the T1 episodes that the planner (fixed rule, clean depth) solves and VLA-0 does not, the share that each
  trained model solves.
- What the VLA adds beyond the demonstrations: a small policy trained from scratch on VLA-P's demonstrations against
  VLA-P on T1 (`small_iid16`). One policy per task, a ResNet-18 image encoder (ImageNet weights) and an MLP
  that gives the next 16 steps' 7 action numbers, as VP-VLA's head does, with an L1 loss; its training settings are
  fixed in a dated note before it is trained. It has no language input, so T2 does not test it.
- Does a VLA read the instruction: VLA-0 and VLA-PL on T1's first 24 episodes of each task with an empty instruction
  and with a wrong one (another task's: the spoon and carrot tasks swap theirs, and so do the stack and eggplant
  tasks), against their own instructions on the same episodes (exact McNemar): `vla31_{blank,wrong}_iid16` and
  `vlapl_{blank,wrong}_iid16`. A VLA that ignores the instruction succeeds as often as with its own.
- A learning curve: VLA-P trained on the first 50 and 125 of its demonstrations of each task (200 and 500 in all),
  with the same settings, on T1 (`vlap200_iid16`, `vlap500_iid16`), each against VLA-0.
- The demonstrations: the pool's success rate of each method and task, the clip report, the targeting scores of the
  chosen and the randomly drawn episodes, and the corrections' takeovers.
- Each eval's cost per episode (`score_costs`).

**GPU hours.** About 120: the demonstrations about 23 (most of it VLA-0's pool episodes, for VLA-S), the 7 training
runs about 45 (the smoke tests measure the speed), the VLA-only evals about 50 and the planner-only evals about 4.

**Built before their first run**, each with its code and tests: the selection of each model's demonstrations as
above, VLA-PLTD's mixture, VLA-0 reading the subtask, the blank and wrong instructions, and the small policy.

**Change of 2026-10-30, before any of the study's demonstrations, training runs or evals.** Noah decided that the
harder scenes are training data too, and that a new test set of harder and more varied scenes takes their place as
the scenes that no model trains on. The plan above stays as it is except for the points below, and `score_marriage`
follows them.

- **The harder scenes in training.** The 5 harder scenes join the training with 250 demonstrations each, as
  SimplerEnv's 4 tasks, so each model's set has 2,250 (9 tasks).
  - The fixed rule cannot read the two-step and two-plate instructions, so the planner reading the VLM's marks
    (`TASK_SOURCE=vp`, clean depth) collects them, on the pool's episodes 0 to 699 of each harder scene
    (`demos_vp_harder_iid100`), then 300 more at a time for a scene with fewer than 250 successes, up to 2,000
    episodes of a scene. Only successful episodes are kept: the first 250 of each scene, in episode order. A scene
    that still has fewer keeps what it has, and the report says so.
  - The visual-prompt labels take each goal of a scene in turn (the two-step task's two). The harder scenes' own
    instructions are training instructions now, and each scene gets 8 training rewordings in each of the four
    families, written after the exclusion list's change of this date (`train_files/rewordings.py`).
  - VLA-P, VLA-PL, VLA-PLT and VLA-PLTD take these demonstrations (with `--reword`, except VLA-P). VLA-PLT targets
    SimplerEnv's 4 tasks only, as before (its ASM has no column for a harder scene), and takes VLA-PL's harder-scene
    demonstrations. VLA-PLTD's corrections stay on SimplerEnv's 4 tasks, as the corrections use the fixed rule.
  - VLA-S: VLA-0 on the harder scenes' pool (`demos_vla_harder_iid100`), episodes 0 to 399 of each scene and then
    100 more at a time until it has 250 successes, at most 600 episodes of a scene. A scene with fewer than 250
    successes after 600 episodes leaves a gap. VLA-0's next successful pool episodes of SimplerEnv's 4 tasks fill it,
    an equal share from each (the first tasks in their order take the remainder), so VLA-S has as many
    demonstrations as VLA-PL.
  - The learning curve takes the first 50 and 125 demonstrations of each of the 9 tasks (450 and 1,125 in all).
  - T3 now tests the harder scenes on new layouts (`iid17`); they are no longer scenes that no model trained on.
    H4 does not change.
- **T5, the new test scenes.** 8 tasks that no model trains on, two of each kind (`eval_files/novel_scenes.py`):
  new objects (a sponge and a soda can, from SimplerEnv's Google-robot objects), new pairs (the eggplant on the
  plate, and the carrot into the basket of the eggplant task's sink scene), reasoning in clutter (the vegetable that
  is not orange, and the one of two carrots that is closer to the towel) and long tasks (a stack built on the plate,
  and three pick-and-places in one instruction). T5 is `iid18`: layout seed 18, a random layout for each episode, 48
  episodes per task (384 episodes). Each task has layouts of its own, so each episode is its own unit. T5's
  instructions and concept words are in the rewordings' exclusion list since its change of this date. The evals
  (with the settings of the evals above, `TASK_SET=novel`): `vla_novel_iid18`, `vlapl_novel_iid18`,
  `pseed31_novel_iid18_none` and `vp_novel_iid18_none`.
- **Planned tests.** A new H5, and the H5 above becomes H6. The order and the rule (a test counts only when every
  test before it was a win) stay.

  | # | Question | Comparison | Test sets |
  |---|---|---|---|
  | H5 | Does the married system beat the current system on scenes that no model trained on? | the married system vs. the current system (H4's) | T5 (384 episodes) |
  | H6 | Does choosing where to collect help? | VLA-PLT vs. VLA-PL | T1 and T2 (784) |

  The systems are H4's, scored the same way, at clean depth. T5's tasks have no column in the ASMs, so on T5 each
  system takes an ASM without task columns (one P(planner) and one P(VLA) for every task), trained on the same
  episodes as its clean-depth ASM. Its per-task choice on T5 is the method with the higher success rate on all of
  those episodes.
- **Smoke tests of T5's scenes** (before any T5 eval, not part of the results, and not on `iid18`): the first 8
  episodes of each task on `iid98`. Their first frames, checked by eye: every object in place and in view, and the
  can upright after settling. The planner reading the VLM's marks on all 8 tasks, and the planner reading the
  simulator's positions (`sim_task.sh` with `--target-source sim`, which takes the first goal) on the 6 one-goal
  tasks: the success check works, and the objects can be picked up and placed. If the planner with the simulator's
  positions cannot pick up and place a task's object in any of the 8 episodes, the task changes in a dated note
  before any T5 eval (for the can: SimplerEnv's opened Red Bull can, which is narrower).
- **GPU hours.** About 160 in all: the harder-scene demonstrations about 25 more (most of it VLA-0's pool episodes,
  for VLA-S), T5's smoke tests and evals about 12, and the training runs about 45 as before (the same number of
  steps for the larger sets).
- **Built before their first run**, as well: the visual-prompt labels of a scene with several goals, the harder
  scenes' training rewordings, and the harder scenes' part of each model's demonstrations.

**Built on 2026-10-30, before any of the study's runs: the selection of each model's demonstrations**
(`train_files/select_demos.py`, with its tests). It reads the five pools (the planner's two, VLA-0's two and the
corrections), each a run folder recorded with `RECORD_DEMOS=1`, and follows the rules above. Each task's episodes must
all be there from episode 0 on, each once and with its pool's layouts. The report gives each task's episodes and
successes, what it needs and the chunks still to run; a set is written only when all of its pools are done.
`planner_eval.sh` now takes chunks (`TASK:START:END`), for the harder scenes' pool, which grows 300 episodes at a time.
Four points that the plan left open are fixed here:

- A task of SimplerEnv with fewer than 250 of the planner's successes in its 700 episodes keeps what it has, as a
  harder scene does, and the report says so. VLA-S takes as many of each task as VLA-PL has of it, so that the two
  sets stay the same size, and the harder scenes' gap is counted against VLA-PL's numbers.
- VLA-PLT's random draw: each task has its own draw with seed 0, from the task's other successes in episode order.
- VLA-PLTD's two sets are written as one set (`targeted_corrections_reworded`, mixture `vla_pltd`). The training draws
  each set of a mixture equally often, whatever its size, so a mixture of the two would give the 1,000 corrections
  half of the draws; one set gives every step the same chance, as DAgger's aggregated data does.
- The learning curve's models are `vlap450` and `vlap1125` (mixtures `vla_p450` and `vla_p1125`), and their evals
  `vlap450_iid16` and `vlap1125_iid16` replace `vlap200_iid16` and `vlap500_iid16`.

**Built on 2026-10-31, before any of the study's runs: the blank and wrong instructions** (`instruction_sets.py`,
`INSTRUCTIONS=blank` or `INSTRUCTIONS=wrong` in the runners, with their tests). `blank` is the empty instruction;
`wrong` gives each of SimplerEnv's 4 tasks another task's own instruction (the spoon and carrot tasks swap theirs, and
so do the stack and eggplant tasks). As a reworded instruction does, either one replaces the task's own everywhere:
VP-VLA's VLM reads it to choose the visual prompts, and the action head reads it. They are not reworded sets, so the
studies of the reworded sets do not see them.

**Built on 2026-10-31, before any of the study's runs: VLA-0 reading the subtask** (`ACTION_INSTRUCTION=subtask` in
`vla_eval.sh`, which passes `--action-instruction subtask`; `instruction_sets.subtask_instruction`, with its test).
When an episode starts, VP-VLA's VLM splits the instruction into subtasks; at the first frame and at each change of
the gripper, it decides whether the current subtask is done. The action head reads the subtask that these decisions
have reached: from the step where the VLM decides that a subtask is done, it reads the next one. VP-VLA's client also
keeps the subtask of the VLM's last question, which changes only at the next question. The head does not read that
one, because in a pick and place it would still be the pick while the robot carries the object; VP-VLA's labeling of
its own training data also moves on at the decision. When the VLM gives no subtasks, the client splits the
instruction by its own rule, and the head reads the instruction; it also reads the instruction in place of a subtask
that the VLM leaves empty. Only the head's text changes: the VLM gets the same
questions as in VLA-0's evals (the whole instruction and the current subtask), so the visual prompts come about as
before. Each text that the head reads is in the run's `sim.log` (`[ACTION TEXT]`). The option needs SAM 3, the VLM
and its subtasks, as every VLA eval has them.

**Built on 2026-10-31, before any of the study's runs: the small policy** (`train_files/small_policy.py`, and
`POLICY=small` in `vla_eval.sh`, with its tests). Its training settings are fixed here, before it is trained:

- One policy for each of SimplerEnv's 4 tasks, trained on VLA-P's demonstrations of that task. Its samples are the
  ones that VLA-P's training draws from them, made by VP-VLA's own training data code from VLA-P's set: the camera
  picture and the picture with the visual prompts (224 x 224), and the scaled action chunk (the next 16 steps' 7
  numbers, in VLA-P's scale). Each sample is made once, and the batches draw from them uniformly, with replacement.
- The model: a ResNet-18 with ImageNet weights (torchvision's `IMAGENET1K_V1`), shared by the two pictures; their
  two sets of 512 features go to an MLP with two hidden layers of 512 (ReLU), which gives the 112 numbers. Every
  weight trains.
- An L1 loss on the 112 numbers; AdamW with learning rate 1e-4 and weight decay 1e-4; 500 warm-up steps, then a
  cosine decay to 0; batches of 64 and 20,000 steps per task (1.28 million draws); bf16; seed 0.
- The only augmentation: both pictures of a sample move by the same random amount, up to 8 pixels each way, with
  the edge pixels repeated. VP-VLA's training data code applies none (its Bridge data config has its picture
  transforms switched off), but a policy trained from scratch on a few hundred demonstrations needs one to be a fair
  baseline. The evals take the pictures as they are.

The eval `small_iid16` is VLA-P's eval of T1 with the small policies in place of VP-VLA's policy server: the same
client, the same visual prompts from SAM 3.1 and the VLM, the same ensembling of the action chunks and the same scale.
The server picks each request's policy by the task's own instruction, which the client sends; the policies read no
language. The smoke run measures the training's speed; the training is short next to the VLAs', and the eval costs
one VLA-only eval of T1.

**Fixed on 2026-11-02, after the smoke tests and before the first model of the study trains: the training settings,
and T5's sponge.** The smoke tests of 2026-11-01 (not part of the results):

- The clip report of the planner's recorded demonstrations (the pool's first 20 episodes of each task; 70 successes,
  3,345 steps): no action number lies outside the checkpoint's range in more than 1% of the steps (0.0% for each of
  the 7), so the planner's steps get no limit. Every converted step has both visual prompts.
- The eggplant's box on rendered frames: the box lies on the inner part of the yellow basket (the simulator's target
  plane in it, with which SimplerEnv's eggplant task checks success), not on the whole basket, and the crosshair is on
  the eggplant in every frame. The carrot task's box covers the plate.
- VLA-0's recording (2 episodes of each task, 7 successes) converts.
- The short training run: 1.87 s per step at a batch of 16, at most 42.5 GB of GPU memory (of 48), 19 GB of files
  per model.

The training settings of the 7 models (the defaults of `run_finetune_1gpu.sh`, Noah's choice): 20,000 steps at a batch
of 16 (320,000 samples, about 3 passes over VLA-P's set), the lowest 24 of the 36 language-model layers frozen, and the
released run's learning rates (1e-4 for the action head, 1e-5 for the VLM) with 1,000 warm-up steps and its cosine
decay, bf16 and its seed. That is about 10.4 GPU hours for each model, 73 in all (not 45), and about 216 for the study.

T5's smoke tests (`iid98`, the first 8 episodes of each task):

- First frames, by eye: every object in place and in view in all 64 episodes, and the can upright in all 8.
- The planner with the simulator's positions: soda can 8/8, carrot by the towel 4/8, carrot in the basket 6/8,
  eggplant on the plate 5/8, the vegetable that is not orange 6/8, sponge 0/8. Each sponge episode failed with "grasp
  missed": 7 cm wide on its narrow side, the sponge does not fit in the open gripper, which only just holds the 6.6 cm
  can. By the rule above, the sponge task changes: the same sponge at 0.6 of its size (4.2 x 6 x 2.2 cm, with
  SimplerEnv's model scale; `novel_scenes.SCALES`), Noah's choice. Its layouts change with its size.
- The planner reading the VLM's marks: soda can 0/8 (its grasps 3 cm off the can's centre, where the open gripper has
  no room to spare; the rule is about the planner with the simulator's positions, so the can stays), carrot by the
  towel 6/8, carrot in the basket 8/8, eggplant on the plate 8/8, the vegetable that is not orange 8/8, sponge 0/8,
  the stack 5/8, the three steps 1/8.

Before any T5 eval, the smoke tests run again with the smaller sponge (the commands above): the planner with the
simulator's positions on its 8 episodes, and the planner reading the VLM's marks on all 8 tasks. Their first frames
are checked by eye. If the planner with the simulator's positions still cannot pick up and place the sponge, the task
changes again in a dated note.

**The rerun on 2026-11-02 (not part of the results).** With the smaller sponge, the planner with the simulator's
positions placed it in 5 of its 8 episodes (before: 0/8), and the planner reading the VLM's marks in 8/8 (before: 0/8).
The other 7 tasks gave the same number of successes as on 2026-11-01. In the first frames, the sponge is in place and
in view in all 8 episodes, about as wide as the carrot. So the sponge task stays as it is.

## VLM-router study

The plan below was fixed on 2026-10-31, before any eval of `iid14` ran, and `score_vlm_router` follows it. The
results do not change it. A result that is not in the plan is reported as exploratory.

**Question.** A router built on a vision-language model (VLM) alone chooses the method for each episode from the
camera picture and the instruction, without any training on the methods' outcomes. Does the two-model ASM, which
predicts each method's success from the planner's first look, choose better than such a VLM router?

**The VLM router** (`vlm_router.py`). On each episode's first frame, the VLM gets the camera picture, the instruction
and a description of the two methods, and it is asked once per method whether that method will complete the
instruction (yes or no). The VLM is the system's own, Qwen3-VL-4B-Instruct, served as for the plan check
(`vlm_server_subtask_simpler.py`, the fast settings). Its probability of "yes" from its first answer token (no
sampling) is the method's score, and the episode goes to the planner when the planner's score is at least the VLA's.
This is the two-model ASM's rule, with the VLM's answers in place of the two success models. Two versions of the
question are asked on every episode:

- **described**: the two methods as their designers would describe them, with what each does badly: the planner's
  depth camera is very noisy, and the VLA is less precise and often fails when the object lies at an unusual angle
  (the full text is in `vlm_router.py`);
- **rates**: the same, and each method's successes on this task in the 100 training episodes of `iid11` with harsh
  depth (spoon: planner 54, VLA 74; carrot: 51 and 63; stack: 54 and 22; eggplant: 46 and 78), as a router that
  remembers past episodes would know them.

The questions were written before the VLM answered any of them, and they are not tuned.

**ASM.** The two-model ASM with the fixed rule, as in the VLM-plan study's fixed-rule rows: trained on
`pseed31_iid11_harsh` and `vla31_iid11` (`selector.train_two_model` with the default features); the planner when
P(planner) >= P(VLA).

**Test set.** `iid14`: 100 episodes of each of SimplerEnv's 4 tasks, each episode index with a random layout of its
own, the tasks' own instructions and harsh depth. It was made as the learned features' second training set (see
"Learned first-frame features"). None of its evals had run when this plan was fixed (the GPU 0 queue runs them
last), and no model in this study is trained on it.

**Evals.** `router_iid14_harsh`: the fixed rule's planner-only eval with the VLM router's questions (harsh depth,
planner seed 0, noise seed 0, SAM 3.1, `VLM_ROUTER=1`). The planner's outcomes, the ASM's first look and the
router's scores all come from it. `vla31_iid14`: the VLA-only eval, from the learned-features data. As a check, the
planner's outcomes are compared with `pseed31_iid14_harsh`, the same episodes and seeds without the questions; they
should agree.

**Planned tests.** On `iid14`, the ASM against (1) the described VLM router and (2) the VLM router with rates: the
exact two-sided paired sign-flip test of the success difference, with one value per episode index (the sum over its
4 tasks, which share their layout). Only the episode indices with all 4 tasks count. The two tests are judged
together by Holm's method at 0.05: the smaller p must be below 0.025, and then the other below 0.05. A win:
significant, with the ASM higher. Intervals: 95% percentile bootstrap of the mean difference per episode, over the
episode indices (10,000 draws).

**What we expect.** The VLM cannot see the depth errors that make the planner fail. The rates version knows what the
per-task choice knows, and the picture adds little that the VLM can judge. So we expect both VLM routers to do no
better than the per-task choice, and the ASM to beat both. The rates version should do better than the described
one, because the rates say which method is better on each task.

**Secondary (reported, no claims).** The success rate of every row, with its interval: the planner alone, the VLA
alone, the ASM, both VLM routers, the per-task choice (each task's better method on `iid11`) and the oracle. Each VLM
router against the better single method and against the per-task choice, the ASM against the per-task choice, and the
rates version against the described one. Each router's share of episodes to the planner, per task, and how often it
chooses as the ASM does. Each router's scores as a success model of its method (AUC and within-task AUC), against the
ASM's P(planner) and P(VLA), with DeLong's test. The VLM's probability of answering yes or no at all (when it is low,
its answers say little).

**Limits.** One VLM (the system's own, a 4B model; a larger VLM might route better), one wording of each question,
harsh depth only, and SimplerEnv's 4 tasks with their own instructions. The router sees the first frame only, as a
VLM router would; the ASM sees the planner's first look, which needs the depth image and SAM. The router eval's
first-look times include the router's questions, so no cost result uses it.

**Cost.** About 30 GPU minutes on GPU 0 for `router_iid14_harsh`: the planner-only eval takes about 15 minutes, and
the questions add 1,600 one-token answers. `vla31_iid14` and `pseed31_iid14_harsh` are in the GPU 0 queue anyway.

## Frontier-model study

The plan below was fixed on 2026-11-09, before any call to the frontier model, before any of the study's evals, and
before `score_marriage` ran (the study's tests use the marriage study's evals, which were not scored yet). The
results do not change it. A result that is not in the plan is reported as exploratory.

**Question.** The marriage study trains a VLA on the planner's episodes. A general frontier model needs no such
training. Two arms ask what it can do in this system:

- **The plan arm**: the frontier model writes the planner's job, and the planner does the motion. Does it give the
  planner a better job than the system's own 4B VLM (F1)? Does it then beat the trained VLA (F2) and the married
  system (F3)?
- **The agent arm**: the frontier model drives the arm itself, with no pick-and-place script. Does the frontier
  model still need the planner (A1)? Does the agent beat the trained VLA (A2) and the married system (A3)?

**The model.** Claude Fable 5.2 (`claude-fable-5-2`), effort `high`, through its API. The API has no temperature
setting, so the same input can give different answers: every episode is run once, its answers are saved, and the
saved answers are the record. No episode is run again, whatever its result. A refusal, or an error that remains
after the client's retries, ends the model's part of the episode: in the plan arm it counts as "not feasible", in
the agent arm as `give_up`. There is no fallback model. The requests hold the simulator's pictures, the
instruction and the tools' answers only. The calls run on a machine of ours that holds the API key; the shared GPU
server holds no key.

**The plan arm.** `TASK_SOURCE=vlm` as it is (see "VLM plan"): one question on each episode's first frame, the same
text (`vlm_plan.prompt`, written for Qwen3-VL, with points in 0-1000 coordinates), the same parser, at most 4 moves,
and the same planner after it. Only the model that answers changes: one request per episode, with the picture and
the question, no tools and no system prompt. The first frames are saved first, the model answers them, and the
planner-only eval reads the saved answers in place of the VLM server. The question is not tuned. A smoke test on
`iid98` (the first 8 episodes of each task; not part of the results) checks the parser and measures the cost. If it
shows a problem of form (not of content), the fix is written in a dated note before the first test call.

**The agent arm.** No fixed rule, no pick-and-place script, no SAM and no VLA: the model decides every motion. It
gets the instruction and these tools, and no state of the simulator:

| Tool | What it does |
|---|---|
| `look` | the camera picture now (the camera that the VLA sees) |
| `locate(x, y)` | the 3D position of a point of the picture, from the depth image (clean depth) and the camera's calibration |
| `move_to(x, y, z, turn)` | moves the gripper, pointing down, to a position with a turn about the vertical axis; the path comes from the planner's motion planning (mplib); it answers whether the motion was done, and the new picture |
| `gripper(open or close)` | opens or closes the gripper, and answers with the new picture |
| `done`, `give_up` | end the model's part of the episode |

An episode has at most 30 turns of the model and the eval's step limit; the motions' steps count as the planner's do.
Success is the simulator's check, as in every eval. Each episode starts a new conversation: the model keeps nothing
from one episode to the next.

The harness (the system prompt, the tools' descriptions and the steps' accounting) is built and changed on `iid98`
only: the first 8 episodes of each of the 17 tasks, and at most 3 passes over them. It is then frozen in a dated
note, with its commit, before the first test episode. That note also gives the cost per episode that the last pass
measured, and Noah's spending limit for the arm. If the test sets would cost more than the limit, the arm runs
whole sets only, in the order T5, T2, T3, and its tests take the sets that ran; this too is fixed in the note,
before the first test episode.

**Test sets.** The marriage study's, with clean depth: T2 (the 4 reworded sets on `paper`, 384 episodes), T3 (the
harder scenes on `iid17`, 240) and T5 (the new scenes on `iid18`, 384): 1,008 episodes. T1 (`iid16`, 400) is
secondary. The episodes, layouts and seeds are those of the existing evals, so every comparison is paired.

**Evals.** All with SAM 3.1 (where the planner reads names), noise seed 0, planner seed 0 and clean depth; `S` is a
reworded set.

| System | T1 | T2 | T3 | T5 |
|---|---|---|---|---|
| The planner reading the frontier plan (new) | `fplan_iid16_none` | `fplan_words_S_paper_none` | `fplan_harder_iid17_none` | `fplan_novel_iid18_none` |
| The planner reading Qwen3-VL's plan (new) | `vlmplan_iid16_none` | `vlmplan_words_S_paper_none` | `vlmplan_harder_iid17_none` | `vlmplan_novel_iid18_none` |
| The frontier agent (new) | | `fagent_words_S_paper` | `fagent_harder_iid17` | `fagent_novel_iid18` |
| The planner reading the VLM's marks | `vp_iid16_none` | `vp_words_S_paper_none` | `vp_harder_iid17_none` | `vp_novel_iid18_none` |
| VLA-PL | `vlapl_iid16` | `vlapl_words_S_paper` | `vlapl_harder_iid17` | `vlapl_novel_iid18` |

The married system is the marriage study's, scored offline by its rules (H4's on T2 and T3, H5's on T5).

**Planned tests.** Two families, one for each arm, each at 0.025 (Bonferroni for the two), so 0.05 for the study.
In a family the order is fixed, each test is the exact two-sided paired sign-flip test, and each counts only when
every test before it in the family was a win (significant, with the frontier arm higher). A significant difference
in the other direction is reported as a loss. All tests take T2, T3 and T5 together (1,008 episodes; for the agent
arm, the sets that ran).

| # | Question | Comparison |
|---|---|---|
| F1 | Is a frontier model a better front-end for the planner than the 4B VLM? | the frontier plan vs. each of Qwen3-VL's plan and the VLM's marks |
| F2 | Does the frontier model with the planner beat the trained VLA? | the frontier plan vs. VLA-PL |
| F3 | Does it beat the married system? | the frontier plan vs. the married system |
| A1 | Does the frontier model need the planner? | the frontier agent vs. the frontier plan |
| A2 | Does the agent beat the trained VLA? | the frontier agent vs. VLA-PL |
| A3 | Does it beat the married system? | the frontier agent vs. the married system |

F1 is an intersection-union test, as the marriage study's H3: a win when the frontier plan is significantly higher
than each of the two; its p is the larger one. The units are the marriage study's (T2: a layout, with one task's
episode in the 4 reworded sets; T3: an episode index, with its 5 scenes; T5: an episode), and only the units that
every eval of a test has in full count. Intervals: 95% percentile bootstrap of the mean difference per episode, over
the units (10,000 draws).

**Secondary (reported, no claims).** Every row's success rate on each set and each task. The frontier plan on T1
with clean and with harsh depth (the answers are the same, because the camera picture is the same; only the
planner's depth changes). How often the frontier plan says "not feasible", and how often its point lies in the SAM
mask that the planner uses. For the agent: turns per episode, the share of episodes that end with `done`,
`give_up` or a limit, and the success rate of each. Refusals, errors, tokens, cost and time per episode, for both
arms.

**What we expect.** The frontier model reads reworded, reasoning and several-step instructions better than a 4B
VLM, so F1 should be a win. F2 and F3 are open: at clean depth the planner is precise, so the plan arm can win.
For the agent, public reports give frontier models high scores on a two-arm benchmark only with a harness made for
them, and much lower scores with a plain one, so A1 is open in both directions; a loss in A1 says that the frontier
model still needs the planner.

**Limits.** One frontier model, one run of each episode, and a model behind an API, which can change under the same
name (the saved answers and their dates are the record). Clean depth only: the plan arm must fall with harsh depth
as the planner does, and a frontier arm with an ASM and a VLA behind it is not in this study. The plan arm sees the
first frame only. The agent's harness is built by us in at most 3 passes on 136 episodes; a better harness may give
a better agent, so the agent arm measures this harness with this model. The harness is written with the help of an
AI assistant of the model's own family; the fixed question, the frozen harness, the saved answers and the paired
tests guard against that.

**Built before their first run.** The saved-answers reader for `TASK_SOURCE=vlm`, the first-frame export, the agent
controller and `score_frontier`. `score_frontier` is checked on made-up data before it reads any result.

**Cost (estimates; the smoke tests measure them).** The plan arm: about 1,550 requests, 50 to 170 USD. The agent
arm: up to 30 requests in each of 1,008 episodes, 500 to 2,500 USD, under the spending limit above. GPU 0: 8
planner-only evals (the frontier plan's and Qwen3-VL's), about 15 to 80 minutes each, and the agent's evals, which
hold the simulator while the model answers; their time is measured on `iid98`.

**Note of 2026-11-13: how the plan arm and the scorer are built.** Written before any call to the frontier model.
Qwen3-VL's plan evals (`vlmplan_*`) ran on that day; no result of theirs was read. The plan above is not changed;
this note says how the code keeps it, where the plan leaves a choice.

- *Three parts* (`planner/frontier.py`). `frames_eval.sh` saves each episode's first frame as a PNG file, with the
  episode, its instruction and the picture's hash. `frontier_answers.py` asks the model on the machine with the key
  and saves each answer as one line of an answers file. `planner_eval.sh` with `VLM_PLAN_ANSWERS` reads the answers
  in place of the VLM server. An answer is tied to its episode, to the question's text and to the picture: the eval
  stops when one of them is not the saved one.
- *One request per episode, once.* The request has the picture and `vlm_plan.prompt` only, with at most 32,000
  output tokens (the model's thinking and its answer together). An episode with a line in the answers file is never
  asked again. Before each request the episode is written to a second file, so an episode whose run was killed
  after the request went out gets an error line at the next run, not a second request.
- *Refusals and errors.* The client retries a request 6 times (connection errors, timeouts, and the answers 408,
  409, 429 and 5xx). A refusal, or an error that remains, gives a line with an empty answer, which the parser reads
  as "not feasible". After an error line the run stops, so that an outage costs one episode and not the rest of
  the set. Before the first episode of a run, one small request without a picture checks the key, the model's name
  and the settings; when it fails, no episode is asked.
- *A loss.* The plan says that a significant difference in the other direction is reported as a loss. For F1, with
  its two comparisons, `score_frontier` reports a win when the frontier plan is significantly higher than each of
  the two, and a loss when it is significantly lower than each of the two; each comparison's own line gives its
  direction.
- *The sets that the agent ran.* The agent arm's tests take the test sets that have all of their agent evals; of
  these, as in every test, only the units that every eval of the test has in full.
- *Wrong joins.* `score_frontier` stops when a record under the frontier plan's names has no saved answer, when a
  record under Qwen3-VL's names has one, or when a record's depth level is not the eval's.

**Note of 2026-11-13: how the agent arm is built.** Written before any call to the frontier model. The harness in
`planner/agent.py` is draft 0: it is not frozen, and the passes on `iid98` may change it (`HARNESS` names its
state, and every episode's record has it with the hashes of the system prompt and of the tools).

- *The controller* (`--controller agent`, `agent_eval.sh`). Each episode is a new conversation: the system prompt,
  the six tools and the instruction. Each turn sends the whole conversation so far, unchanged, and the model's
  answer is kept as it came. The model calls one tool in a turn; an answer without a tool call costs its turn. A
  wrong input (a pixel outside the picture, a turn of more than 90 degrees) is answered as not done and costs its
  turn.
- *The tools.* `look`: the picture that the VLA would get, as a PNG file. `locate`: the position of a pixel's
  surface in the robot's base frame, from the clean depth. `move_to`: a move of 10 cm or less goes in a straight
  line, as the planner's own short moves do; a longer move is planned by mplib (seeded per episode, planner seed 0)
  around what the camera sees, without the arm and, with the gripper closed, without what it holds. It answers
  that the motion was done when the hand ends within 1.5 cm of the position. `gripper`: 5 steps, as the planner's.
  The motions' steps are the planner's (at most 2 cm in a step, and up to 3 steps that wait for the hand).
- *The ends.* `done`, `give_up`, the turn limit (30 turns) or the eval's step limit. A refusal ends the model's part
  as `give_up`. An answer that is cut off, or that stops in a way the harness does not know, also ends it as
  `give_up`, with the reason in the record.
- *The key stays on our machine.* The controller sends each turn's request to the relay (`agent_relay.py`) through
  an SSH tunnel; the relay adds nothing and changes nothing, and it refuses a request for another model or another
  effort. Its log has each request's tokens and cost, and no content.
- *Errors.* The relay's client retries a request 6 times. An error that remains ends the model's part of the
  episode as `give_up`. The eval then stops before its next episode, so that an outage costs one episode in each
  running task; the other episodes are started again as chunks (`TASK:START:END`), and no episode is run twice.
- *The record.* Each episode: one line in `agent_stats.jsonl` (the end, the turns, the tools that were called, the
  tokens, the cost at the list prices and the model's time) and the conversation as it was sent, with every picture
  as a PNG file (`transcripts/`). The runner's `status.log` has no success rates.

## Instruction study

The plan below was fixed on 2026-11-13, after the marriage study was scored and before any of this study's code,
smoke tests or evals. The results do not change it. A result that is not in the plan is reported as exploratory.

**Why.** Two results of the marriage study. With an empty or a wrong instruction, VLA-PL succeeds on T1 about as
often as with its own (76.0% and 78.1% against 79.2%), because each of SimplerEnv's scenes has one task; so the
reworded sets cannot show whether a fine-tuned VLA follows the instruction. And an exploratory count on T3, made
after the scoring: in the kitchen scene, whose spoon and carrot tasks share their layouts, VLA-PL solves both tasks
in 35 of 48 layouts (VLA-0 in 16, the planner in 44 with the fixed rule and in 45 with the VLM's marks), so there it
does follow the instruction, but both tasks are in its training set. No eval so far asks for a goal that the scene
allows and that no model trained on.

**Question.** In one scene, with the same layouts, does a controller do what the instruction says when the
instruction pairs an object with a target that it was never paired with? Is the fine-tuned VLA tied to the pairings
of its training set (I1), more than the released VLA (I2), and does the planner then do better than the fine-tuned
VLA (I3)?

**Tasks** (`TASK_SET=choice`, `eval_files/choice_scenes.py`). The kitchen scene of `harder_scenes.py`: the spoon, the
towel, the carrot and the plate, with its random layouts (`random_layout`), so episode N of the 4 tasks has the same
objects in the same places. Every instruction has the form that the fixed rule reads.

| Task | Instruction | Goal | Pairing |
|---|---|---|---|
| `ChoiceSpoonOnTowel-v0` | put the spoon on the towel | the spoon on the towel | trained (SimplerEnv's spoon task, and the kitchen scene's) |
| `ChoiceCarrotOnPlate-v0` | put carrot on plate | the carrot on the plate | trained (SimplerEnv's carrot task, and the kitchen scene's) |
| `ChoiceSpoonOnPlate-v0` | put the spoon on the plate | the spoon on the plate | new: in no training set of this repo |
| `ChoiceCarrotOnTowel-v0` | put the carrot on the towel | the carrot on the towel | new: in no training set of this repo |

The goals are checked as in the harder scenes: the spoon as SimplerEnv's spoon goal (part of it may lie off the
target), the carrot as SimplerEnv's carrot goal (all of it on the target, touching nothing else but the robot). Each
task also logs, without counting it, whether the object lies on the other target at the end; every eval of this
study, the VLA-only ones too, writes the goal and the logged pair of each episode to its episode records. 120 steps.

**Layout set.** `iid19`: layout seed 19, a random layout for each episode, 48 episodes per task (192 episodes for a
controller). No eval has run on it. The unit of every test is an episode index (48 units), because its 4 tasks share
the layout.

**Controllers.** Those of the marriage study, with its settings (SAM 3.1, noise seed 0, planner seed 0, clean
depth; the VLAs with the fast settings and VLM seed offset 0): VLA-0 (`vla_choice_iid19`), VLA-PL
(`vlapl_choice_iid19`), the planner with the fixed rule (`pseed31_choice_iid19_none`) and the planner reading the
VLM's marks (`vp_choice_iid19_none`). The frontier-model study's arms are not part of this study.

**Planned tests.** Three tests, judged together by Holm's method at 0.05; each is the exact two-sided paired
sign-flip test with one value per episode index. A unit's *drop* for a controller is its successes on the two
trained pairings minus its successes on the two new pairings (from -2 to 2).

| # | Question | Value per unit | Shown when |
|---|---|---|---|
| I1 | Is the fine-tuned VLA tied to its trained pairings? | VLA-PL's drop | significant, above 0 |
| I2 | Is it tied to them more than the released VLA? | VLA-PL's drop minus VLA-0's drop | significant, above 0 |
| I3 | On a new pairing, does the planner do better than the fine-tuned VLA? | the successes on the two new pairings of the planner reading the VLM's marks, minus VLA-PL's | significant, above 0 |

A significant difference in the other direction is reported as such. Intervals: 95% percentile bootstrap over the
episode indices (10,000 draws).

**Secondary (reported, no claims).** Each controller's success rate on each task. Each controller's *following
score*: of the 96 pairs of an episode index and an object, the share where it solves both of the object's tasks (a
controller that does the same thing whatever the instruction says can solve only one of the two, because the object
ends on one target). On the new pairings, how often the object ends on its trained target instead (the logged
pair). The fixed rule against the VLM's marks. VLA-0's drop alone.

**What we expect.** Both planners solve the four tasks at about their rate in the kitchen scene of T3 (above 90%),
because the instructions name the object and the target plainly. VLA-PL solves the trained pairings at about its T3
rate (about 80%) and far fewer of the new ones, often with the object on its trained target, so I1 and I3 are shown.
VLA-0 is weak in this scene on every task, so I2 is open: a small drop from a low level may not differ from a large
drop from a high one.

**Smoke tests first** (not part of the results, and not on `iid19`): the first 8 episodes of each task on `iid98`.
The first frames are checked by eye. The planner reading the simulator's positions (`--target-source sim`) must pick
up and place each task's object in at least one of the 8 episodes; if it cannot for a task, the task's goal check
changes in a dated note before any eval of the study.

**Built before their first run.** `choice_scenes.py` and its tests, `TASK_SET=choice` in the runners, the goal and
the logged pair in every eval's episode records, and `score_instruction`, which is checked on made-up data before
it reads any result.

**Limits.** One scene with two objects and two targets, 48 layouts, and instructions in the fixed rule's form: the
study does not test rewording, look-alike targets (T3's two plates do) or more than one move. VLA-PL trained in this
scene (500 of its 2,006 episodes are the kitchen scene's trained pairings, and 250 more its two-step task) and VLA-0
did not, so I2 compares a model that knows the scene with one that does not. A drop can come from the motion to an
unfamiliar target as well as from the instruction; the logged pair separates the two only in part. VP-VLA's own training data (real robot episodes) may hold such pairings: new means new for the fine-tuning.

**Cost.** About 4 GPU hours on GPU 0: two VLA-only evals of 192 episodes and two planner-only evals, and the smoke
tests.
