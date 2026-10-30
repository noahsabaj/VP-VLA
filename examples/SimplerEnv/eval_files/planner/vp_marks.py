"""VP-VLA's VLM marks: the VLM that gives VP-VLA its visual prompts, not the fixed rule, tells the planner what to do.

VP-VLA's VLM (vlm_server_subtask_simpler.py) splits the instruction into subtasks and, for the current subtask, names
the object to move and, in a placing subtask, the target; VP-VLA's SAM turns these names into its crosshair (the
object) and its box (the target). With `--task-source vp` the planner asks the same VLM the same questions on the
first frame, one for each subtask (VP-VLA asks about the current subtask only, at the first frame and at each gripper
change; the planner plans from its first look, so it asks about every subtask there), and runs its pick-and-place
script once per move found. An instruction with no move goes to the VLA in the selector mode.

Needs nothing but the standard library; the VLM server answers (messages "task_decomposition" and
"subtask_detection").
"""

import re
from typing import List, Optional, Tuple

# VP-VLA's subtasks when its VLM gives none (parse_explicit_task in model2simpler_interface_visual_prompt.py).
SUBTASK_SPLIT = r",\s*and\s*|,\s*|\s+and\s+"


def subtasks(vlm, instruction: str) -> Tuple[List[str], str]:
    """The instruction's subtasks as VP-VLA makes them with --vlm-decompose-subtasks, and where they come from: the
    VLM's ("vlm"), else VP-VLA's split ("rule")."""
    parts = vlm.decompose_task_to_subtasks(instruction)
    if parts:
        return list(parts), "vlm"
    return [p.strip() for p in re.split(SUBTASK_SPLIT, instruction.lower()) if p.strip()] or [instruction], "rule"


def mark(name) -> Optional[str]:
    """A name from the VLM's answer (target_object or target_location); None when it names nothing."""
    name = str(name or "").strip()
    return None if name.lower() in ("", "null", "none") else name


def steps(marks: List[dict]) -> List[dict]:
    """The pick-and-place moves from the marks, one {"object", "target"} per subtask in order: each subtask that names
    a target ends a move of the object that it names (else of the last object named before it). Of two moves of the
    same object in a row only the later one is kept: a pick subtask that names a target too only looks ahead to its
    place subtask. The moves have the VLM plan's form (vlm_plan.parse), without points."""
    moves, last = [], None
    for m in marks:
        last = m["object"] or last
        if not (m["target"] and last):
            continue
        move = {"object": last, "object_point": None, "target": m["target"], "target_point": None}
        if moves and moves[-1]["object"].lower() == last.lower():
            moves[-1] = move
        else:
            moves.append(move)
    return moves


def ask(vlm, instruction: str, image) -> dict:
    """VP-VLA's VLM marks on the first frame (uint8 RGB): {"subtasks", "subtasks_from", "marks", "steps"}, with a
    mark {"subtask", "object", "target", "text"} for each subtask. Each question is VP-VLA's at a first frame (no
    earlier frame, the gripper open), so the one about the first subtask is the one VP-VLA asks itself. Raises when
    the server fails."""
    parts, source = subtasks(vlm, instruction)
    marks = []
    for i, part in enumerate(parts):
        answer = vlm.detect_subtask(reference_image=None, current_image=image, task_description=instruction,
                                    current_subtask=part, next_subtask=parts[i + 1] if i + 1 < len(parts) else None,
                                    gripper_state="OPEN")
        if answer.get("error"):
            raise RuntimeError(f"VLM subtask detection failed: {answer['error']}")
        marks.append({"subtask": part, "object": mark(answer.get("target_object")),
                      "target": mark(answer.get("target_location")), "text": answer.get("raw_response")})
    return {"subtasks": parts, "subtasks_from": source, "marks": marks, "steps": steps(marks)}
