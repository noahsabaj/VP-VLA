"""CPU tests for the VLM plan (planner/vlm_plan.py): the prompt, parsing the VLM's answer, and choosing a mask."""

import numpy as np

from examples.SimplerEnv.eval_files.planner import vlm_plan


def test_prompt():
    p = vlm_plan.prompt("put the spoon on the towel and the carrot on the plate")
    assert '"put the spoon on the towel and the carrot on the plate"' in p
    assert '"feasible": false' in p and "0-1000" in p and f"at most {vlm_plan.MAX_STEPS}" in p


def test_parse_two_steps():
    text = """```json
{"feasible": true, "steps": [
  {"object": "the Spoon", "object_point": [500, 250], "target": "towel", "target_point": [1000, 0]},
  {"object": "carrot", "object_point": [10, 2000], "target": "plate", "target_point": "near the robot"}]}
```"""
    plan = vlm_plan.parse(text, width=641, height=481)
    assert plan["feasible"] and plan["error"] is None and len(plan["steps"]) == 2
    first, second = plan["steps"]
    assert (first["object"], first["target"]) == ("spoon", "towel")
    assert first["object_point"] == [320.0, 120.0] and first["target_point"] == [640.0, 0.0]
    assert second["object_point"] is None and second["target_point"] is None  # out of range, not a point


def test_parse_not_feasible():
    assert vlm_plan.parse('{"feasible": false, "steps": []}', 640, 480) == {
        "feasible": False,
        "steps": [],
        "error": None,
    }
    for text, error in [
        ("I cannot do that.", "no JSON"),
        ("{feasible: true}", "bad JSON"),
        ('{"feasible": "yes", "steps": []}', "no feasible"),
        ('{"feasible": true, "steps": []}', "no steps"),
        ('{"feasible": true, "steps": [{"object": "carrot"}]}', "no object or no target"),
        ('{"feasible": true, "steps": [' + ", ".join(['{"object": "a", "target": "b"}'] * 5) + "]}", "more than"),
    ]:
        plan = vlm_plan.parse(text, 640, 480)
        assert not plan["feasible"] and not plan["steps"] and error in plan["error"], (text, plan)


def test_pick_mask():
    masks = np.zeros((3, 20, 30), bool)
    masks[0, 2:6, 2:6] = True  # best score, far left
    masks[1, 10:15, 20:25] = True
    masks[2, 12:18, 20:28] = True  # overlaps mask 1 at the point below
    scores = [0.9, 0.4, 0.6]
    assert vlm_plan.pick_mask(masks, scores, None) == 0  # no point: the best score, as with the fixed rule
    assert vlm_plan.pick_mask(masks, scores, [22.0, 13.0]) == 2  # inside 1 and 2: the better score of the two
    assert vlm_plan.pick_mask(masks, scores, [21.0, 11.0]) == 1  # inside 1 only
    assert vlm_plan.pick_mask(masks, scores, [8.0, 3.0]) == 0  # inside none: the nearest
    assert vlm_plan.contains(masks[1], [20.4, 10.4]) and not vlm_plan.contains(masks[1], [-1.0, 10.0])
