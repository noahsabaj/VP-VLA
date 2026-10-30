"""CPU tests for VP-VLA's VLM marks as the planner's job (planner/vp_marks.py; planner_client.py, task_source vp)."""

import numpy as np
import pytest

from examples.SimplerEnv.eval_files.planner import selector as s
from examples.SimplerEnv.eval_files.planner import vp_marks as vp

IMAGE = np.zeros((4, 4, 3), np.uint8)


class FakeVLM:
    """VP-VLA's VLM server: the same subtasks for every instruction, and for each subtask its (object, target)."""

    def __init__(self, subtasks, marks, error=None):
        self.subtasks, self.marks, self.error, self.questions = subtasks, marks, error, []

    def decompose_task_to_subtasks(self, instruction):
        return list(self.subtasks)

    def detect_subtask(self, **question):
        self.questions.append(question)
        obj, target = self.marks.get(question["current_subtask"], ("", None))
        answer = {"decision": "continue", "target_object": obj, "target_location": target, "raw_response": "{}"}
        return {**answer, "error": self.error} if self.error else answer


def _marks(*pairs):
    return [{"subtask": str(i), "object": o, "target": t} for i, (o, t) in enumerate(pairs)]


def _moves(*pairs):
    return [{"object": o, "object_point": None, "target": t, "target_point": None} for o, t in pairs]


def test_each_subtask_with_a_target_ends_a_move():
    assert vp.steps(_marks(("carrot", None), ("carrot", "plate"))) == _moves(("carrot", "plate"))
    assert vp.steps(_marks(("carrot", None), (None, "plate"))) == _moves(("carrot", "plate"))  # the last object named
    two = _marks(("carrot", None), ("carrot", "plate"), ("spoon", None), ("spoon", "towel"))
    assert vp.steps(two) == _moves(("carrot", "plate"), ("spoon", "towel"))
    assert vp.steps(_marks(("carrot", None), ("spoon", None))) == []
    assert vp.steps(_marks((None, "plate"))) == []  # a target, but no object named yet


def test_a_pick_subtask_that_names_a_target_only_looks_ahead():
    assert vp.steps(_marks(("spoon", "towel"), ("spoon", "towel"))) == _moves(("spoon", "towel"))
    assert vp.steps(_marks(("Carrot", "table"), ("carrot", "plate surface"))) == _moves(("carrot", "plate surface"))


def test_a_mark_is_a_name_or_none():
    assert [vp.mark(n) for n in (None, "", " ", "null", "None", " carrot ")] == [None] * 5 + ["carrot"]


def test_the_planner_asks_about_every_subtask_on_the_first_frame():
    vlm = FakeVLM(
        ["pick up the carrot", "place the carrot on the plate"],
        {"pick up the carrot": ("carrot", None), "place the carrot on the plate": ("carrot", "plate")},
    )
    plan = vp.ask(vlm, "the plate should get the carrot", IMAGE)
    assert plan["subtasks_from"] == "vlm" and plan["steps"] == _moves(("carrot", "plate"))
    assert [(m["subtask"], m["object"], m["target"]) for m in plan["marks"]] == [
        ("pick up the carrot", "carrot", None),
        ("place the carrot on the plate", "carrot", "plate"),
    ]
    first, second = vlm.questions
    assert first.pop("current_image") is IMAGE
    # VP-VLA's question at a first frame (no earlier frame, the gripper open): the first one is VP-VLA's own
    assert first == {
        "reference_image": None,
        "task_description": "the plate should get the carrot",
        "current_subtask": "pick up the carrot",
        "next_subtask": "place the carrot on the plate",
        "gripper_state": "OPEN",
    }
    assert second["current_subtask"] == "place the carrot on the plate" and second["next_subtask"] is None


def test_vp_vlas_split_when_the_vlm_gives_no_subtasks():
    vlm = FakeVLM([], {"put carrot on plate": ("carrot", "plate"), "the spoon on the towel": ("spoon", "towel")})
    plan = vp.ask(vlm, "Put carrot on plate, and the spoon on the towel", IMAGE)
    assert plan["subtasks"] == ["put carrot on plate", "the spoon on the towel"] and plan["subtasks_from"] == "rule"
    assert plan["steps"] == _moves(("carrot", "plate"), ("spoon", "towel"))


def test_a_server_error_stops_the_eval():
    with pytest.raises(RuntimeError, match="subtask detection failed: busy"):
        vp.ask(FakeVLM(["put carrot on plate"], {}, error="busy"), "put carrot on plate", IMAGE)


def test_no_target_gives_the_episode_to_the_vla():
    gated = {"choice": "vla", "reason": "VLM marks: no target"}
    assert s.vlm_plan_gate({"task_source": "vp", "vlm_plan_feasible": False}) == gated
    assert s.vlm_plan_gate({"task_source": "vp", "vlm_plan_feasible": True}) is None
    assert s.vlm_plan_gate({"task_source": "rule"}) is None
    assert s.vlm_plan_gate({"task_source": "vlm"})["reason"] == "VLM plan: not pick-and-place"


def test_the_client_takes_its_moves_from_the_marks():
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from examples.SimplerEnv.eval_files.planner.planner_client import PlannerClient

    c = PlannerClient.__new__(PlannerClient)  # no servers: only what the first look reads before the camera
    c.task_description, c.steps, c.stats = "put carrot on plate", None, {}
    c.vlm_planner = FakeVLM(["put carrot on plate"], {"put carrot on plate": ("carrot", "plate")})
    assert c._vp_plan(IMAGE) and c.steps == _moves(("carrot", "plate"))
    assert c.stats["vlm_plan_feasible"] and c.stats["steps_done"] == 0 and c.stats["vlm_plan_sam"] == []
    assert c.stats["vp_subtasks"] == ["put carrot on plate"] and c.stats["vp_subtasks_from"] == "vlm"

    # No target: the planner-only try ends at the first look (in the selector mode the gate gives it to the VLA).
    c.task_description, c.steps, c.stats, c.first_look = "open the drawer", None, {}, True
    c.task_source, c.hand_off, c.step_count = "vp", False, 1
    c.vlm_planner = FakeVLM(["open the drawer"], {"open the drawer": ("drawer", None)})
    c._approach(IMAGE, None, None, None, None)
    assert c.phase == "done" and c.stats["failure"] == "VLM marks: no target" and c.steps is None
    assert s.vlm_plan_gate({**c.stats, "task_source": "vp"})["reason"] == "VLM marks: no target"
