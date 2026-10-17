"""The VLM plan check: before the planner moves, the VLM sees what the planner chose and says whether that is enough.

The planner reads every instruction with one fixed rule and then runs one pick-and-place script, so it fails when the
rule misreads the instruction (a reworded instruction), when SAM3 finds the wrong object (two plates), or when one
pick-and-place cannot finish the task (two steps). On the first frame, the planner marks on the camera picture the
object it will pick up (tinted green) and the target it will put it on (a blue box) and asks the VLM whether doing
exactly that finishes the instruction. The VLM's probability of "yes" (from its first answer token, no sampling) is
the feature `vlm_plan_yes`; below THRESHOLD the episode goes to the VLA (planner_client.py in the selector mode;
`gate` here for the offline scores). Needs only numpy; the VLM server answers (vlm_server_subtask_simpler.py,
message "yes_no").
"""

from typing import Optional

import numpy as np

THRESHOLD = 0.5
OBJECT_COLOR = np.array([0, 255, 0])
TARGET_COLOR = np.array([0, 120, 255])


def mark_plan(image: np.ndarray, grasp_mask: np.ndarray, place_mask: np.ndarray, alpha: float = 0.45,
              border: int = 3) -> np.ndarray:
    """The camera picture (H, W, 3 uint8) with the object to pick up tinted green and a blue box around the target."""
    out = image.astype(np.float32).copy()
    g = np.asarray(grasp_mask).astype(bool)
    out[g] = (1 - alpha) * out[g] + alpha * OBJECT_COLOR
    ys, xs = np.nonzero(np.asarray(place_mask))
    if len(ys):
        h, w = out.shape[:2]
        y0, y1 = max(ys.min() - border, 0), min(ys.max() + border, h - 1)
        x0, x1 = max(xs.min() - border, 0), min(xs.max() + border, w - 1)
        out[y0:y0 + border, x0:x1 + 1] = TARGET_COLOR
        out[y1 - border + 1:y1 + 1, x0:x1 + 1] = TARGET_COLOR
        out[y0:y1 + 1, x0:x0 + border] = TARGET_COLOR
        out[y0:y1 + 1, x1 - border + 1:x1 + 1] = TARGET_COLOR
    return out.round().clip(0, 255).astype(np.uint8)


def question(instruction: str) -> str:
    """The question the VLM gets, with the camera picture and the marked picture."""
    return (f'The robot is told: "{instruction}"\n'
            "The first image is the robot's camera view. The second image is the same view with a motion planner's "
            "plan marked on it: the planner will pick up the object tinted green, put it down on the place inside "
            "the blue box, and then stop. It will do nothing else.\n"
            "Will doing exactly this complete the whole instruction? Answer yes or no.")


def gate(p_yes: Optional[float]) -> str:
    """The controller for an episode: the VLA when the VLM doubts the planner's plan, else the planner."""
    if p_yes is None:
        raise ValueError("no plan check in this record (run the planner with --plan-check-port)")
    return "vla" if p_yes < THRESHOLD else "planner"
