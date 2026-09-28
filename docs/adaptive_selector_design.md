# Adaptive selector: planner first, VLA as fallback

Status: draft, 2026-09-28. Follows KAN-17 (motion planner in SimplerEnv).

## Idea

For each try, a small learned model (the **selector**) looks at the scene before the robot
moves and predicts whether the motion planner's pick-and-place will succeed. If it is
confident, the planner does the whole task. If not, the VLA does the whole task from its
normal start pose. There is no mid-task hand-off, so the VLA never starts from a pose it
wasn't trained on, which KAN-21 showed costs it a few points.

This is the "adaptive selection module" box in Dr. Zhu's diagram: VLM, then selector, then
VLA or motion planner.

## What KAN-17 tells us (SimplerEnv, 4 tasks x 24 layouts)

| | Spoon | Carrot | Stack | Eggplant | Avg |
|---|---|---|---|---|---|
| VLA alone (KAN-16, 5 runs) | 60.8 | 45.0 | 20.8 | 87.5 | 53.5 |
| MPlib planner alone | 100 | 91.7 | 66.7 | 95.8 | 88.5 |
| Perfect selector (best of the two per layout) | 100 | 95.0 | 70.8 | 100 | 91.5 |

- The planner fails on 11 of 96 layouts. The VLA succeeds on only about 3 of those, so a perfect
  selector adds just 3 points here. SimplerEnv is nearly solved by the planner.
- 11 failures are too few labels to train anything. We need more, and harder, layouts.
- The planner's advantage comes partly from perfect simulator depth. A fair comparison with
  VP-VLA must degrade that (see "Honest comparison" below).

So the selector only shows its value where the planner struggles and the VLA doesn't. The plan
builds those conditions in on purpose, and measures them.

## Selector

**When it runs:** once, on the first frame, before any motion. It is cheap: SAM3 plus one plan
attempt (20 ms for MPlib) are already computed there.

**Inputs (features), all things a real robot would have:**
- Perception: the SAM3 score for the object and the target, mask size in pixels, the number of valid
  depth points on the object, the object's height and long and short extents, and how spread out
  the depth points are.
- Planning: whether the plan succeeded, the planned path length, the planning time, and the
  closest distance to an obstacle along the path.
- Grasp geometry: how much the hand must turn, and the object width against the gripper opening.
- Task: which task it is (one-hot), and later a text embedding of the instruction for unseen tasks.
- VLA confidence: the spread of the action chunk, if we switch to a head that gives one. QwenOFT's
  head is deterministic, see codebase notes.

**Output:** P(planner succeeds). Choose the planner if P is above a threshold t, otherwise the VLA.
The threshold is picked on held-out data.

**Model:** start with logistic regression and a small gradient-boosted tree. With hundreds of
examples, a neural net isn't justified yet. Report both, plus two baselines: "always planner" and a
hand-written rule (plan failed or SAM3 score < 0.3 means VLA).

**Labels:** run both controllers on the same layouts. The label for the planner is its success.
The VLA's success on that layout (averaged over runs) gives the value of falling back. The metric
is the success of the chosen controller, compared with always-planner, always-VLA and the
perfect selector.

## Getting enough, and hard enough, data

1. **More layouts:** SimplerEnv's `--obj-variation-mode xy` places objects on a grid
   (`--obj-init-x/y`). A 10 x 10 grid per task gives 400 layouts instead of 96. The planner-only
   runs are cheap (about 4 minutes per 24 tries). VLA runs cost about 50 minutes per 24 tries on
   one GPU, so VLA labels are the bottleneck.
2. **Sensor noise (the honest comparison):** add a depth-noise option to the planner's camera
   view, off by default:
   - Gaussian noise that grows with distance (like a RealSense: sigma about 1 mm at 0.5 m, rising
     with depth squared),
   - random dropped pixels and holes at object edges,
   - a small camera pose error (a few mm and a fraction of a degree).

   Sweep three levels: none, realistic and harsh. The planner should degrade and the VLA
   (colour only) shouldn't, which gives the selector real work.
3. **Harder scenes, later:** distractor objects, clutter near the target, and objects that are
   hard for a top-down grasp (the spoon on its edge, the stacked cubes). The Stack task is already
   where the planner is weakest.

## Honest comparison with the paper

- Headline table: VP-VLA (paper and ours), planner alone, and planner + selector. Each is
  reported with clean depth and realistic depth. The realistic-depth row is the one to compare
  with the paper.
- Also say plainly what the planner gets that the VLA doesn't: depth, camera calibration and
  joint angles. The robot mask currently comes from the simulator. On a real robot it would come
  from rendering the arm at its known joint angles, so switch to that before any headline claim.
- Use the same 24 layouts per task as the paper for the headline numbers, and the grid layouts
  only for training the selector. Never test on training layouts.

## Build order

1. Log the selector features from every planner try (planner-only runs are cheap to redo).
2. Add the depth-noise option and re-run planner-only at 3 noise levels on the 96 layouts.
3. Generate VLA labels on the same layouts (the costly step; share GPUs with the SAM3.1 work).
4. Train v0 (logistic regression) with leave-one-task-out and held-out layouts. Report against the
   baselines and the perfect selector.
5. Add `--controller selector`: run the selector on the first frame, then hand the whole
   try to the planner or the VLA.
6. Evaluate on the 96 headline layouts, with clean and realistic depth.

## Open questions

- Should the selector also choose between MPlib and cuRobo? They were close in KAN-21 (MPlib is
  faster and simpler), so probably keep MPlib only.
- Should it be allowed to switch mid-try, for example when a grasp fails? That's a later
  version. v0 decides once.
