"""The VLM plan: the VLM, not the fixed rule, tells the planner what to do.

The planner's fixed rule (sam3_client.extract_target_from_task and planner_client.place_target_from_task) reads one
object and one target from an instruction of the form "put X on Y". With `--task-source vlm` the VLM reads the
instruction and the first camera frame instead and writes the planner's job: whether the instruction can be done
with pick-and-place moves only, and if so the moves in order, each an object and a target with a short name for SAM3
and a point in the picture. The planner runs its pick-and-place script once per move; the points choose between
look-alike objects (two plates). An instruction that is not pick-and-place goes to the VLA in the selector mode.

Needs only numpy; the VLM server answers (vlm_server_subtask_simpler.py, message "generate", greedy decoding).
"""

import json
import re
from typing import List, Optional

import numpy as np

MAX_STEPS = 4
POINT_SCALE = 1000.0  # Qwen3-VL gives image points in 0-1000 coordinates


def prompt(instruction: str) -> str:
    return f"""The robot arm can only do pick-and-place moves. One move: pick up one object, put it on or in one \
target, and let go.

Instruction: "{instruction}"

Look at the picture. Can the instruction be done with pick-and-place moves only, one after another? If yes, list the \
moves in order, at most {MAX_STEPS}. For each move give the object to pick up and the target to put it on or in: a \
short name (one or two words, such as "carrot" or "yellow basket") and a point on it in the picture, [x, y] in \
0-1000 coordinates. When two objects look alike, the point tells them apart.

If the instruction cannot be done with pick-and-place moves only, answer {{"feasible": false, "steps": []}}.

OUTPUT JSON ONLY:
{{"feasible": true, "steps": [{{"object": "<name>", "object_point": [x, y], "target": "<name>", \
"target_point": [x, y]}}]}}"""


def _point(value, width: int, height: int) -> Optional[List[float]]:
    """A 0-1000 point as pixels (x, y), or None if it is not two numbers in range."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not (0 <= x <= POINT_SCALE and 0 <= y <= POINT_SCALE):
        return None
    return [x * (width - 1) / POINT_SCALE, y * (height - 1) / POINT_SCALE]


def parse(text: str, width: int, height: int) -> dict:
    """The VLM's answer as {"feasible", "steps", "error"}: steps are dicts with object, target (lower-case names) and
    object_point, target_point (pixels, or None). An answer that does not parse, or a feasible one without usable
    steps, counts as not feasible, with the reason in "error"."""
    def no(error: Optional[str]) -> dict:
        return {"feasible": False, "steps": [], "error": error}

    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return no("no JSON in the answer")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        return no(f"bad JSON: {e}")
    if not isinstance(data, dict):
        return no("the JSON is not an object")
    if data.get("feasible") is not True:
        return no(None if data.get("feasible") is False else "no feasible: true or false")
    raw = data.get("steps")
    if not isinstance(raw, list) or not raw:
        return no("feasible but no steps")
    if len(raw) > MAX_STEPS:
        return no(f"{len(raw)} steps, more than {MAX_STEPS}")
    steps = []
    for s in raw:
        if not isinstance(s, dict):
            return no("a step is not an object")
        names = [re.sub(r"^(the|a|an) ", "", str(s.get(k) or "").strip().lower()) for k in ("object", "target")]
        if not all(names):
            return no("a step has no object or no target")
        steps.append({"object": names[0], "target": names[1],
                      "object_point": _point(s.get("object_point"), width, height),
                      "target_point": _point(s.get("target_point"), width, height)})
    return {"feasible": True, "steps": steps, "error": None}


def pick_mask(masks, scores, point: Optional[List[float]]) -> int:
    """Which SAM3 mask is meant: without a point the best-scoring one (as with the fixed rule); with a point the
    best-scoring mask that contains it, else the mask nearest to it."""
    scores = np.asarray(scores, dtype=float)
    if point is None:
        return int(np.argmax(scores))
    inside = [i for i in range(len(masks)) if contains(masks[i], point)]
    if inside:
        return max(inside, key=lambda i: scores[i])
    dist = []
    for m in masks:
        ys, xs = np.nonzero(np.asarray(m))
        dist.append(np.min(np.hypot(xs - point[0], ys - point[1])) if len(xs) else np.inf)
    return int(np.argmin(dist))


def contains(mask, point: List[float]) -> bool:
    m = np.asarray(mask)
    x, y = round(point[0]), round(point[1])
    return 0 <= y < m.shape[0] and 0 <= x < m.shape[1] and bool(m[y, x])
