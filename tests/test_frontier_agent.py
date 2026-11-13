"""CPU tests for the frontier-model study's agent arm: the conversation and the controller (planner/agent.py, with a
made-up model and a made-up robot: no request leaves the machine), the relay for the machine that holds the key
(experiments/agent_relay.py) and the scorer reading the controller's records."""

import copy
import json
import socket
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from examples.SimplerEnv.eval_files import novel_scenes as ns
from examples.SimplerEnv.eval_files.planner import agent, frontier
from examples.SimplerEnv.eval_files.planner.experiments import agent_relay
from examples.SimplerEnv.eval_files.planner.experiments import score_frontier as sf

README = Path(__file__).parents[1] / "examples/SimplerEnv/eval_files/planner/experiments/README.md"
TASK = next(iter(ns.TASKS))
KEY = f"{TASK}/e98/0"
INSTRUCTION = "put the sponge on the plate"
IMAGE = np.random.default_rng(0).integers(0, 256, (48, 64, 3), dtype=np.uint8)
USAGE = {"input_tokens": 1000, "output_tokens": 100, "cache_creation_input_tokens": 500, "cache_read_input_tokens": 2000}
INFO = {"env_name": TASK, "obj_episode_id": 0, "layout_seed": 98, "layout_per_episode": True, "steps": 7}


def message(*calls, stop_reason=None, text=None):
    """The model's answer as the relay gives it: a thinking block, then text or tool calls (name, input)."""
    content = [{"type": "thinking", "thinking": "", "signature": "made-up"}]
    if text is not None:
        content.append({"type": "text", "text": text})
    content += [{"type": "tool_use", "id": f"toolu_{id(c)}", "name": c[0], "input": c[1]} for c in calls]
    return {
        "id": "msg_made_up",
        "model": frontier.MODEL,
        "role": "assistant",
        "content": content,
        "usage": USAGE,
        "stop_reason": stop_reason or ("tool_use" if calls else "end_turn"),
    }


class Model:
    """A made-up model: the answers of `script` in order (a ModelError is raised), then `then` for ever. It keeps a
    copy of each request."""

    def __init__(self, *script, then=("done", {})):
        self.script, self.then, self.requests = list(script), then, []

    def __call__(self, request):
        self.requests.append(copy.deepcopy(request))
        answer = self.script.pop(0) if self.script else self.then
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, dict) else message(answer)


class Recorded:
    """A model with a copy of each request that it got."""

    def __init__(self, ask):
        self.ask, self.requests = ask, []

    def __call__(self, request):
        self.requests.append(copy.deepcopy(request))
        return self.ask(request)


class FakeRobot:
    """The simulator's side: the hand goes where it is told (unless `stuck`), and every command is kept."""

    def __init__(self, stuck=False, no_path=False):
        self.stuck, self.no_path = stuck, no_path
        self.position, self.started, self.commands, self.moves = np.zeros(3), [], [], []

    def start(self, obs, env, key):
        self.started.append(key)

    def locate(self, obs, x, y):
        return None if x == 0 else (0.3, 0.1, 0.02)

    def plan_move(self, obs, xyz, turn, closed):
        goal = np.r_[xyz, 1.0, 0.0, 0.0, 0.0]
        self.moves.append((tuple(xyz), turn, closed))
        if self.no_path:
            return "no path to the position was found (made up)", [], goal
        return None, [np.r_[0.5 * np.asarray(xyz), goal[3:]], goal], goal

    def distance(self, obs, goal):
        return float(np.linalg.norm(self.position - goal[:3]))

    def action(self, obs, desired, gripper, terminate=False):
        if desired is not None and not self.stuck:
            self.position = np.asarray(desired[:3], dtype=float).copy()
        self.commands.append(("stay" if desired is None else "move", gripper, terminate))
        return {}, {"terminate_episode": np.array([1.0 if terminate else 0.0])}


def run(controller, key=KEY, limit=120, info=INFO, success=True):
    """One episode as the evaluator runs it: steps until the controller ends it or the step limit; its steps."""
    controller.set_episode(key)
    controller.reset(INSTRUCTION)
    steps = 0
    while steps < limit:
        _, action = controller.step(IMAGE, INSTRUCTION, obs={}, env=object())
        steps += 1
        if action["terminate_episode"][0] > 0:
            break
    controller.finish_episode(success, {**info, "steps": steps})
    return steps


def make(tmp_path, model, robot=None, **kwargs):
    robot = robot or FakeRobot()
    return agent.AgentController(robot, model, str(tmp_path / TASK), **kwargs), robot


def record(tmp_path, n=-1):
    return json.loads((tmp_path / TASK / frontier.AGENT_FILE).read_text().splitlines()[n])


def results(request, n=-1):
    """The tool results of a request's n-th message: (text, has a picture, is an error) of each."""
    out = []
    for block in request["messages"][n]["content"]:
        content = block.get("content", [])
        out.append(
            (
                " ".join(b["text"] for b in content if b["type"] == "text"),
                any(b["type"] == "image" for b in content),
                bool(block.get("is_error")),
            )
        )
    return out


def test_the_harness_is_the_plans():
    assert agent.MAX_TURNS == 30
    assert [t["name"] for t in agent.TOOLS] == ["look", "locate", "move_to", "gripper", "done", "give_up"]
    for tool in agent.TOOLS:  # strict tools: the model's inputs follow the schema
        assert tool["strict"] is True and tool["input_schema"]["additionalProperties"] is False
        assert set(tool["input_schema"].get("required", [])) == set(tool["input_schema"]["properties"])
    assert agent.TOOLS[2]["input_schema"]["required"] == ["x", "y", "z", "turn"]
    assert agent.TOOLS[3]["input_schema"]["properties"]["state"]["enum"] == ["open", "close"]
    conversation = agent.Conversation(Model(), agent.system_prompt(640, 480))
    conversation.start(INSTRUCTION)
    request = conversation.request()
    assert set(request) == {"model", "max_tokens", "output_config", "system", "tools", "tool_choice",
                            "cache_control", "messages"}  # fmt: skip
    assert request["model"] == "claude-fable-5-2" and request["output_config"] == {"effort": "high"}
    assert request["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}  # no forced tool
    assert request["tools"] is agent.TOOLS and "640 x 480 pixels" in request["system"]
    assert "at most 30 turns" in request["system"] and "SAM" not in request["system"]
    assert request["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": f"The instruction: {INSTRUCTION}"}]}
    ]


def test_a_whole_episode_with_the_scripted_model(tmp_path):
    model = Recorded(agent.scripted_reach())
    controller, robot = make(tmp_path, model)
    steps = run(controller)
    r = record(tmp_path)
    assert r["end"] == "done" and r["turns"] == 7 and r["success"] is True and r["episode"] == KEY
    assert r["tools"] == {"look": 1, "locate": 1, "move_to": 2, "gripper": 2, "done": 1}
    assert (r["refusal"], r["error"], r["obj_episode_id"], r["env_name"]) == (False, None, 0, TASK)
    assert r["harness"] == agent.HARNESS and r["request"]["max_turns"] == 30
    assert robot.started == [KEY]
    # 10 cm above the located point, then 5 cm straight down, the gripper open during both
    assert [m[0] for m in robot.moves] == [pytest.approx((0.3, 0.1, 0.12)), pytest.approx((0.3, 0.1, 0.07))]
    assert [m[1:] for m in robot.moves] == [(0.0, False), (0.0, False)]
    # the steps: 2 poses for each move, 5 for each of the gripper's changes, and the last one that ends the episode
    assert steps == 2 + 5 + 5 + 2 + 1 == r["steps"]
    closed = [c for c in robot.commands if c[1] == agent.GRIPPER_CLOSED]
    assert len(closed) == 5 and all(c[0] == "stay" for c in closed)
    assert robot.commands[-1] == ("stay", agent.GRIPPER_OPEN, True)
    # the history only grows: each request is the one before it with the model's answer and its result at the end
    for before, after in zip(model.requests, model.requests[1:], strict=False):
        assert after["messages"][: len(before["messages"])] == before["messages"]
        assert len(after["messages"]) == len(before["messages"]) + 2
        assert (after["system"], after["tools"]) == (before["system"], before["tools"])
    last = model.requests[-1]
    assert [results(last, n)[0] for n in (2, 4, 6, 8, 10, 12)] == [
        ("", True, False),
        ("x = 0.300, y = 0.100, z = 0.020 (meters)", False, False),
        ("The motion was done.", True, False),
        ("The gripper is closed.", True, False),
        ("The gripper is open.", True, False),
        ("The motion was done.", True, False),
    ]
    # the saved conversation: the messages as they were sent, with each picture as a PNG file
    saved = json.loads((tmp_path / TASK / "transcripts" / (KEY.replace("/", "__") + ".json")).read_text())
    assert saved["record"] == r and saved["system"] == last["system"] and len(saved["turns"]) == 7
    pictures = sorted((tmp_path / TASK / "transcripts" / KEY.replace("/", "__")).iterdir())
    assert len(pictures) == 5 and np.array_equal(frontier.read_png(str(pictures[0])), IMAGE)
    assert saved["messages"][2]["content"][0]["content"] == [
        {"type": "image", "file": f"{pictures[0].parent.name}/0.png"}
    ]
    assert "base64" not in json.dumps(saved)


def test_tokens_cost_and_time_are_summed(tmp_path):
    controller, _ = make(tmp_path, Model("look", "look"))
    run(controller)
    r = record(tmp_path)
    assert r["turns"] == 3 and r["usage"] == {k: 3 * v for k, v in USAGE.items()}
    assert r["cost_usd"] == pytest.approx(3 * (1000 * 8 + 100 * 40 + 500 * 10 + 2000 * 0.2) / 1e6)
    assert r["model_seconds"] >= 0 and r["models"] == [frontier.MODEL]


def test_a_motion_that_fails_is_answered_as_not_done(tmp_path):
    move = ("move_to", {"x": 0.3, "y": 0.0, "z": 0.1, "turn": 30})
    model = Model(move)
    controller, robot = make(tmp_path, model, FakeRobot(stuck=True))
    steps = run(controller)
    assert steps == 2 + agent.SETTLE_STEPS + 1  # its 2 poses, the steps that wait for the hand, the end
    assert results(model.requests[1]) == [
        ("The motion was not done: the gripper stopped before it reached the position.", True, True)
    ]
    model = Model(move)
    controller, robot = make(tmp_path, model, FakeRobot(no_path=True))
    assert run(controller, key=f"{TASK}/e98/1") == 1  # no motion at all
    assert results(model.requests[1]) == [
        ("The motion was not done: no path to the position was found (made up).", True, True)
    ]
    assert robot.moves == [((0.3, 0.0, 0.1), 30.0, False)]


def test_wrong_inputs_are_answered_and_cost_a_turn(tmp_path):
    model = Model(
        ("locate", {"x": 64, "y": 0}),  # outside the 64 x 48 picture
        ("locate", {"x": 0, "y": 0}),  # no depth there
        ("move_to", {"x": 0.3, "y": 0.0, "z": 0.1, "turn": 120}),
        ("move_to", {"x": "far", "y": 0.0, "z": 0.1, "turn": 0}),
        ("gripper", {"state": "half"}),
        ("fly", {}),
    )
    controller, robot = make(tmp_path, model)
    assert run(controller) == 1 and not robot.moves
    assert [results(model.requests[-1], n)[0] for n in (2, 4, 6, 8, 10, 12)] == [
        ("Not done: `x` must be from 0 to 63.", False, True),
        ("Not done: the camera has no depth at the pixel (0, 0).", False, True),
        ("Not done: `turn` must be from -90 to 90.", False, True),
        ("Not done: `x` must be a number.", False, True),
        ("Not done: `state` must be open or close.", False, True),
        ("Not done: there is no tool `fly`.", False, True),
    ]
    assert record(tmp_path)["turns"] == 7 and record(tmp_path)["end"] == "done"


def test_an_answer_without_a_tool_call_costs_its_turn(tmp_path):
    model = Model(message(text="I will look first."), "look")
    controller, _ = make(tmp_path, model)
    run(controller)
    assert model.requests[1]["messages"][-1] == {"role": "user", "content": [{"type": "text", "text": agent.ONE_TOOL}]}
    assert model.requests[1]["messages"][-2]["content"][0]["type"] == "thinking"  # the answer as it came
    assert record(tmp_path)["turns"] == 3


def test_two_calls_in_one_answer(tmp_path):
    model = Model(message(("look", {}), ("gripper", {"state": "close"})))
    controller, robot = make(tmp_path, model)
    assert run(controller) == 1  # the gripper's call is not done
    assert results(model.requests[1]) == [("", True, False), ("Not done: call one tool in each turn.", False, True)]
    assert all(c[1] == agent.GRIPPER_OPEN for c in robot.commands)


def test_the_turn_limit_and_the_step_limit(tmp_path):
    model = Model(then="look")
    controller, _ = make(tmp_path, model)
    assert run(controller) == 1 and len(model.requests) == 30
    assert (record(tmp_path)["end"], record(tmp_path)["turns"]) == ("turn limit", 30)
    # the eval's step limit ends an episode whose model goes on: a gripper that opens and closes for ever
    model = Model(then=("gripper", {"state": "close"}))
    controller, _ = make(tmp_path, model)
    assert run(controller, key=f"{TASK}/e98/1", limit=12, success=False) == 12
    r = record(tmp_path)
    assert (r["end"], r["turns"], r["success"]) == ("step limit", 3, False)


def test_give_up_a_refusal_and_an_answer_that_was_cut_off(tmp_path):
    controller, _ = make(tmp_path, Model(("give_up", {})))
    run(controller, success=False)
    assert (record(tmp_path)["end"], record(tmp_path)["refusal"], record(tmp_path)["error"]) == ("give_up", False, None)
    controller, _ = make(tmp_path, Model(message(stop_reason="refusal", text="")))
    run(controller, key=f"{TASK}/e98/1", success=False)
    assert (record(tmp_path)["end"], record(tmp_path)["refusal"], record(tmp_path)["turns"]) == ("give_up", True, 1)
    controller, _ = make(tmp_path, Model(message(("look", {}), stop_reason="max_tokens")))
    run(controller, key=f"{TASK}/e98/2", success=False)
    r = record(tmp_path)
    assert r["end"] == "give_up" and r["error"] == "error: the answer stopped with 'max_tokens' and 1 tool calls"
    run(controller, key=f"{TASK}/e98/3")  # neither stops the eval: the next episode runs
    assert record(tmp_path)["end"] == "done"


def test_an_outage_ends_one_episode_and_stops_the_eval(tmp_path):
    model = Model("look", agent.ModelError("502: APIStatusError: overloaded"))
    controller, _ = make(tmp_path, model)
    run(controller, success=False)
    r = record(tmp_path)
    assert (r["end"], r["turns"], r["error"]) == ("give_up", 2, "error: 502: APIStatusError: overloaded")
    controller.set_episode(f"{TASK}/e98/1")
    with pytest.raises(SystemExit, match="the eval stops"):
        controller.reset(INSTRUCTION)
    started = (tmp_path / TASK / "agent_started.jsonl").read_text().splitlines()
    assert [json.loads(line)["key"] for line in started] == [KEY]


def test_the_scorer_reads_the_controllers_records(tmp_path, monkeypatch):
    name = sf.agent_name("novel", "iid18")
    root = tmp_path / name
    controller = agent.AgentController(FakeRobot(), Model(), str(root / TASK))
    run(controller, key=f"{TASK}/e18/0")
    chunk = agent.AgentController(FakeRobot(), Model(("give_up", {})), str(root / TASK / "eps_1_2"))
    run(chunk, key=f"{TASK}/e18/1", info={**INFO, "obj_episode_id": 1}, success=False)
    records = sf.load_agent(str(root), list(ns.TASKS))
    assert {k: (r["end"], r["success"]) for k, r in records.items()} == {
        (TASK, 0): ("done", True),
        (TASK, 1): ("give_up", False),
    }
    assert all(r["end"] in frontier.AGENT_ENDS for r in records.values())
    monkeypatch.setenv("RUNS_DIR", str(tmp_path))
    fr = sf.Frontier()
    assert fr.agent_parts() == (["T5"], [])
    assert fr.row(sf.AGENT, ["T5"])[0] == {("T5", "novel", TASK, 0): 1.0, ("T5", "novel", TASK, 1): 0.0}


# The relay ----------------------------------------------------------------------------------------------------


class ApiError(Exception):
    pass


@pytest.fixture
def relay(tmp_path):
    """A relay with a made-up model on a free port of this machine; (the relay, its address, the model's requests)."""
    requests = []

    def ask(request):
        requests.append(request)
        if request["messages"][-1]["content"] == "fail":
            raise ApiError("overloaded")
        return message(("look", {}))

    made = agent_relay.Relay(ask, (ApiError,), str(tmp_path / "relay.jsonl"))
    server = agent_relay.serve(made, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield made, f"127.0.0.1:{server.server_address[1]}", requests
    server.shutdown()
    server.server_close()


def _request(content="hello", **changes):
    return {"model": frontier.MODEL, "max_tokens": 100, "output_config": {"effort": frontier.EFFORT},
            "messages": [{"role": "user", "content": content}], **changes}  # fmt: skip


def test_the_relay_passes_the_request_and_the_answer_unchanged(relay, tmp_path):
    made, address, requests = relay
    ask = agent.HttpRelay(address, tries=1)
    for _ in range(2):
        answer = ask(_request())
        assert answer["stop_reason"] == "tool_use" and answer["usage"] == USAGE
        assert [b["type"] for b in answer["content"]] == ["thinking", "tool_use"] and answer["content"][1][
            "name"
        ] == "look"
    assert requests[0] == _request()
    rows = [json.loads(line) for line in (tmp_path / "relay.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["usage"] == USAGE and rows[0]["turn"] == 1 and rows[0]["model"] == frontier.MODEL
    assert rows[0]["cost_usd"] == pytest.approx((1000 * 8 + 100 * 40 + 500 * 10 + 2000 * 0.2) / 1e6)
    assert "hello" not in json.dumps(rows) and "messages" not in rows[0]  # the log has no content
    assert made.spent == pytest.approx(2 * rows[0]["cost_usd"]) == pytest.approx(agent_relay.spent_usd(made.log_path))


def test_the_relay_refuses_what_is_not_the_studys(relay):
    made, address, requests = relay
    ask = agent.HttpRelay(address, tries=1)
    with pytest.raises(agent.ModelError, match="400: the model must be claude-fable-5-2, not 'claude-opus-5-5'"):
        ask(_request(model="claude-opus-5-5"))
    with pytest.raises(agent.ModelError, match="400: the effort must be high"):
        ask(_request(output_config={"effort": "low"}))
    with pytest.raises(agent.ModelError, match="400: the request is not a Messages request"):
        ask({"model": frontier.MODEL})
    assert not requests
    with pytest.raises(agent.ModelError, match="502: ApiError: overloaded"):  # after the client's retries
        ask(_request("fail"))
    made.limit_usd = 0.01
    ask(_request())
    with pytest.raises(agent.ModelError, match=r"402: the relay's spending limit of 0\.01 USD is reached"):
        ask(_request())
    assert len(requests) == 2


def test_a_relay_that_cannot_be_reached(tmp_path):
    with socket.socket() as s:  # a port of this machine that nothing listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ask = agent.HttpRelay(f"127.0.0.1:{port}", tries=2, wait=0.0)
    with pytest.raises(agent.ModelError, match="cannot be reached"):
        ask(_request())
    if hasattr(socket, "AF_UNIX") and sys.platform != "win32":
        with pytest.raises(agent.ModelError, match="cannot be reached"):
            agent.HttpRelay(f"unix:{tmp_path}/no.sock", tries=1)(_request())


# The simulator's side (needs OpenCV and SciPy; the motion planner itself runs on the GPU server only) -----------------


def _robot():
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from examples.SimplerEnv.eval_files.planner import geometry, planner_client, planners

    robot = agent.Robot.__new__(agent.Robot)  # no mplib here: the straight moves need none
    robot.geometry, robot.client, robot.planners = geometry, planner_client, planners
    robot.max_step_pos, robot.max_step_rot, robot.home = 0.02, 0.15, np.array([1.0, 0.0, 0.0, 0.0])
    return robot


def _obs(target=(0.3, 0.0, 0.2)):
    """An observation with the camera at the base frame's origin, looking along z, and a flat depth of 0.5 m."""
    depth = np.full((48, 64, 1), 0.5)
    depth[0, 0] = 0.0
    camera = {"rgb": IMAGE, "depth": depth, "Segmentation": np.zeros((48, 64, 2), dtype=int)}
    intrinsic = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]])
    pose = np.array([*target, 1.0, 0.0, 0.0, 0.0])
    return {"image": {"3rd_view_camera": camera},
            "camera_param": {"3rd_view_camera": {"intrinsic_cv": intrinsic, "extrinsic_cv": np.eye(4)}},
            "agent": {"base_pose": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), "qpos": np.zeros(8),
                      "controller": {"arm": {"target_pose": pose}}},
            "extra": {"tcp_pose": pose}}  # fmt: skip


def test_locate_reads_the_depth_and_the_calibration():
    robot = _robot()
    assert robot.locate(_obs(), 32, 24) == pytest.approx((0.0, 0.0, 0.5))
    assert robot.locate(_obs(), 52, 14) == pytest.approx((0.1, -0.05, 0.5))  # (u - cx) z / fx, (v - cy) z / fy
    assert robot.locate(_obs(), 0, 0) is None and robot.locate(_obs(), 64, 0) is None  # no depth; not in the picture


def test_a_short_move_goes_straight_and_a_position_outside_the_workspace_is_refused():
    robot = _robot()
    obs = _obs()
    problem, queue, goal = robot.plan_move(obs, [0.3, 0.0, 0.14], 0.0, closed=False)
    assert problem is None and len(queue) == 3 and goal == pytest.approx([0.3, 0.0, 0.14, 1.0, 0.0, 0.0, 0.0])
    assert [q[2] for q in queue] == pytest.approx([0.18, 0.16, 0.14])  # 2 cm in each step
    problem, queue, _ = robot.plan_move(obs, [0.3, 0.0, 0.9], 0.0, closed=False)
    assert problem == "the position is outside the robot's workspace" and queue == []
    turned = robot.orientation(90.0)  # a quarter turn about the vertical axis
    assert turned == pytest.approx([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    _, action = robot.action(obs, queue[0] if queue else np.r_[0.3, 0.0, 0.18, robot.home], agent.GRIPPER_CLOSED)
    assert action["world_vector"] == pytest.approx([0.0, 0.0, -0.02]) and action["gripper"][0] == -1.0
    assert action["terminate_episode"][0] == 0.0
    _, action = robot.action(obs, None, agent.GRIPPER_OPEN, terminate=True)
    assert not action["world_vector"].any() and action["terminate_episode"][0] == 1.0
    assert robot.distance(obs, goal) == pytest.approx(0.06)


def test_the_readme_has_the_agent_arms_commands():
    text = README.read_text(encoding="utf-8")
    block = text[text.index("## Evals (GPU)") : text.index("## Result files (CPU)")]
    for needle in ("agent_relay ", "AGENT_RELAY=", "AGENT_MODEL=scripted", " fagent_novel_iid18 ",
                   " fagent_harder_iid17 ", " fagent_words_${set}_paper "):  # fmt: skip
        assert needle in block, needle
