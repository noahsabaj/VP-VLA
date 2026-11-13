"""The frontier-model study's agent arm (experiments/README.md, "Frontier-model study").

The frontier model drives the arm itself: no fixed rule, no pick-and-place script, no SAM and no VLA. It gets the
instruction and the tools of TOOLS, and no state of the simulator. An episode has at most MAX_TURNS turns of the
model (one request each) and the eval's step limit; each episode is a new conversation.

- Conversation: the model's side. Each turn sends the whole conversation so far, unchanged (the history only grows),
  and reads one tool call from the answer. A refusal, or an error that remains after the client's retries, ends the
  model's part of the episode as give_up.
- AgentController: the evaluator's model interface (`--controller agent`). It does the tool calls: `look` and
  `locate` at once, `move_to` and `gripper` over the next simulator steps, and it answers them with the new picture.
- Robot: the simulator's side of the tools (the camera's depth, mplib, the hand's commands). Its imports need the
  simulator's environment, so the rest of this module, and its tests, do without it.
- HttpRelay: the model through the relay on the machine that holds the API key (experiments/agent_relay.py), reached
  through an SSH tunnel; the GPU server holds no key. scripted_reach: a made-up model for pipeline checks, which
  sends no request to anything.

The harness is SYSTEM, TOOLS and the accounting constants below. It is changed on iid98 only, in at most 3 passes,
and then frozen in a dated note (HARNESS says which state this is).
"""

import base64
import http.client
import json
import os
import re
import socket
import time
import zlib
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import frontier

HARNESS = "draft 0 (2026-11-13), before the first pass on iid98"
MAX_TURNS = 30  # the model's turns in an episode
MAX_TOKENS = 16000  # of one turn: the model's thinking and its tool call
HOLD_STEPS = 5  # simulator steps that the gripper takes to open or close, as in the planner
SETTLE_STEPS = 3  # more steps on a motion's last waypoint, until the hand is there, as in the planner
REACHED = 0.015  # m: a motion was done when the hand ends this close to the position
STRAIGHT = 0.10  # m: a move this short goes in a straight line; a longer one is planned around what the camera sees
MAX_TURN_DEG = 90.0
GRIPPER_OPEN, GRIPPER_CLOSED = 1.0, -1.0

SYSTEM = """You control a robot arm on a table through tools. You get one instruction. Do what it says, then call \
`done`.

The robot is a small arm with a two-finger gripper that always points straight down. A fixed camera looks at the \
table and the arm; `look` gives its picture. A picture is {width} x {height} pixels, and a pixel is (x, y) with \
(0, 0) at the top left.

Positions are in meters in the robot's frame: x points forward from the robot's base, y to its left, z up. \
`locate` gives the position of the surface that a pixel shows. `move_to` moves the point between the gripper's \
fingertips to a position; `turn` turns the gripper about the vertical axis, in degrees from its orientation at the \
start. `gripper` opens or closes the fingers; the gripper is open at the start.

A move of {straight_cm:.0f} cm or less goes in a straight line. A longer move is planned around everything that \
the camera sees, so it fails when it ends on or next to an object: end a long move above the object, then go \
straight down.

You have at most {max_turns} turns: each of your replies is one turn, and each must call exactly one tool. The \
robot also has a limited number of motion steps (one step moves the gripper about 2 cm; opening or closing takes \
{hold_steps} steps), so do not waste motion. Call `done` when the instruction is done, or `give_up` when you \
cannot do it."""

NO_INPUT = {"type": "object", "properties": {}, "additionalProperties": False}
NUMBER = {"type": "number"}
TOOLS = [
    {"name": "look", "description": "The camera's picture now. Call it to see the scene.", "strict": True,
     "input_schema": NO_INPUT},
    {"name": "locate",
     "description": "The 3D position, in meters in the robot's frame, of the surface that one pixel of the current "
                    "picture shows. Call it to learn where an object or a place is before you move there.",
     "strict": True,
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["x", "y"], "properties": {
         "x": {"type": "integer", "description": "The pixel's column, from 0 at the left."},
         "y": {"type": "integer", "description": "The pixel's row, from 0 at the top."}}}},
    {"name": "move_to",
     "description": "Moves the point between the gripper's fingertips to a position, with the gripper pointing "
                    "down. It answers whether the motion was done, with the picture after it.",
     "strict": True,
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["x", "y", "z", "turn"],
                      "properties": {
                          "x": {**NUMBER, "description": "Meters forward from the robot's base."},
                          "y": {**NUMBER, "description": "Meters to the robot's left."},
                          "z": {**NUMBER, "description": "Meters up."},
                          "turn": {**NUMBER, "description": "Degrees about the vertical axis from the gripper's "
                                                            "orientation at the start, from -90 to 90."}}}},
    {"name": "gripper", "description": "Opens or closes the gripper's fingers. It answers with the picture after it.",
     "strict": True,
     "input_schema": {"type": "object", "additionalProperties": False, "required": ["state"], "properties": {
         "state": {"type": "string", "enum": ["open", "close"]}}}},
    {"name": "done", "description": "Call it when the instruction is done. It ends the episode.", "strict": True,
     "input_schema": NO_INPUT},
    {"name": "give_up", "description": "Call it when you cannot do the instruction. It ends the episode.",
     "strict": True, "input_schema": NO_INPUT},
]
ONE_TOOL = "Call one tool. When the instruction is done, call `done`; when you cannot do it, call `give_up`."
ENDS = {"done": "done", "give_up": "give_up"}  # the tools that end the model's part, and the end's name


class ModelError(Exception):
    """No answer from the model: an error that remains after the client's retries."""


def system_prompt(width: int, height: int, max_turns: int = MAX_TURNS) -> str:
    return SYSTEM.format(width=width, height=height, max_turns=max_turns, straight_cm=100 * STRAIGHT,
                         hold_steps=HOLD_STEPS)


def text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def picture_block(image: np.ndarray) -> dict:
    """The camera's picture as the model gets it: the PNG file of its pixels."""
    data = base64.standard_b64encode(frontier.png_bytes(image)).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


class Conversation:
    """One episode's conversation with the model. `ask(request)` gives the model's message as a dict (the SDK's
    Message.to_dict()) or raises ModelError."""

    def __init__(self, ask: Callable[[dict], dict], system: str, max_turns: int = MAX_TURNS):
        self.ask, self.system, self.max_turns = ask, system, max_turns
        self.messages: List[dict] = []
        self.turns = 0
        self.records: List[dict] = []  # each turn: its time, why it stopped, its tokens
        self.usage: Counter = Counter()
        self.model_seconds = 0.0
        self.refusal = False
        self.error: Optional[str] = None
        self.outage = False  # the error is the client's (no answer came), not the answer's form
        self.calls: List[dict] = []  # the last answer's tool calls that have no result yet

    def start(self, instruction: str) -> None:
        self.messages.append({"role": "user", "content": [text_block(f"The instruction: {instruction}")]})

    def request(self) -> dict:
        """A turn's request: the same system prompt and tools every time, and the conversation so far. The top-level
        cache_control lets the API reuse the unchanged start of the conversation; it changes nothing that the model
        gets."""
        return {"model": frontier.MODEL, "max_tokens": MAX_TOKENS, "output_config": {"effort": frontier.EFFORT},
                "system": self.system, "tools": TOOLS,
                "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
                "cache_control": {"type": "ephemeral"}, "messages": self.messages}

    def turn(self) -> Tuple[str, ...]:
        """The model's next tool call, ("call", name, input), or the end of its part, ("end", kind). An answer
        without a tool call costs its turn and gets ONE_TOOL as the reply."""
        if self.calls:
            raise RuntimeError("the last tool call has no answer yet")
        while True:
            if self.turns >= self.max_turns:
                return "end", "turn limit"
            self.turns += 1
            t0 = time.perf_counter()
            try:
                message = self.ask(self.request())
            except ModelError as e:
                self.model_seconds += time.perf_counter() - t0
                self.error, self.outage = f"error: {e}"[:500], True
                self.records.append({"turn": self.turns, "seconds": time.perf_counter() - t0, "error": self.error})
                return "end", "give_up"
            seconds = time.perf_counter() - t0
            self.model_seconds += seconds
            usage = {f: (message.get("usage") or {}).get(f) or 0 for f in frontier.USAGE_FIELDS}
            self.usage.update(usage)
            stop, content = message.get("stop_reason"), message.get("content") or []
            self.records.append({"turn": self.turns, "seconds": seconds, "stop_reason": stop, "usage": usage,
                                 "id": message.get("id"), "model": message.get("model")})
            if stop == "refusal":
                self.refusal = True
                return "end", "give_up"
            calls = [b for b in content if b.get("type") == "tool_use"]
            if stop not in ("tool_use", "end_turn") or not content or (stop == "tool_use" and not calls):
                self.error = f"error: the answer stopped with {stop!r} and {len(calls)} tool calls"
                return "end", "give_up"
            self.messages.append({"role": "assistant", "content": content})  # as it came, thinking blocks and all
            if not calls:
                self.messages.append({"role": "user", "content": [text_block(ONE_TOOL)]})
                continue
            self.calls = calls
            return "call", calls[0]["name"], calls[0].get("input") or {}

    def answer(self, blocks: List[dict], is_error: bool = False) -> None:
        """The result of the tool call that turn() gave. A second call in the same answer is not done."""
        results = [{"type": "tool_result", "tool_use_id": self.calls[0]["id"], "content": blocks,
                    **({"is_error": True} if is_error else {})}]
        results += [{"type": "tool_result", "tool_use_id": c["id"], "is_error": True,
                     "content": [text_block("Not done: call one tool in each turn.")]} for c in self.calls[1:]]
        self.messages.append({"role": "user", "content": results})
        self.calls = []


def number(args: dict, name: str, low: float, high: float) -> float:
    """A tool input as a number in a range; ValueError says what is wrong, as the model's answer."""
    value = args.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"`{name}` must be a number")
    if not low <= value <= high:
        raise ValueError(f"`{name}` must be from {low:g} to {high:g}")
    return float(value)


class AgentController:
    """The frontier agent as a controller (the evaluator's model interface). `robot`: the simulator's side of the
    tools (Robot). `ask`: the model (Conversation). In `out_dir` (a task's run folder): one line per episode in
    frontier.AGENT_FILE, and each episode's conversation in transcripts/."""

    def __init__(self, robot, ask: Callable[[dict], dict], out_dir: str, max_turns: int = MAX_TURNS) -> None:
        self.robot, self.ask, self.out_dir, self.max_turns = robot, ask, out_dir, max_turns
        self.key: Optional[str] = None
        self.last_error: Optional[str] = None
        self._clear("")

    def set_episode(self, key: str) -> None:
        self.key = key

    def _clear(self, instruction: str) -> None:
        self.instruction = instruction
        self.conversation: Optional[Conversation] = None
        self.end: Optional[str] = None  # how the model's part ended (frontier.AGENT_ENDS); None while it goes on
        self.queue: List[np.ndarray] = []  # the hand's poses of the motion that is being done
        self.goal: Optional[np.ndarray] = None  # that motion's last pose
        self.settled = 0
        self.hold = 0  # steps left of the gripper's opening or closing
        self.pending: Optional[str] = None  # the tool call whose motion is being done
        self.gripper = GRIPPER_OPEN
        self.tools: Counter = Counter()

    def reset(self, task_description: str) -> None:
        if self.last_error is not None:  # an outage costs one episode, not the rest of the set
            raise SystemExit(f"the last episode's model part ended with an error ({self.last_error}): the eval stops "
                             "here; start its other episodes again when the model answers")
        self._clear(task_description)

    def step(self, image: np.ndarray, task_description: Optional[str] = None, obs=None, env=None, info=None,
             **kwargs):
        if obs is None or env is None:
            raise ValueError("AgentController needs obs and env; run it with planner_evaluator")
        if self.conversation is None:
            if self.key is None:
                raise ValueError("the agent needs the episode key (planner_evaluator sets it)")
            self.robot.start(obs, env, self.key)
            self.conversation = Conversation(self.ask, system_prompt(image.shape[1], image.shape[0], self.max_turns),
                                             self.max_turns)
            self.conversation.start(self.instruction if task_description is None else task_description)
            os.makedirs(self.out_dir, exist_ok=True)
            with open(os.path.join(self.out_dir, "agent_started.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": self.key}) + "\n")
        if self.end is not None:
            return self.robot.action(obs, None, self.gripper, terminate=True)
        if self.queue:
            return self.robot.action(obs, self.queue.pop(0), self.gripper)
        if self.goal is not None and self.settled < SETTLE_STEPS and self.robot.distance(obs, self.goal) > 0.01:
            self.settled += 1
            return self.robot.action(obs, self.goal, self.gripper)
        if self.hold > 0:
            self.hold -= 1
            return self.robot.action(obs, None, self.gripper)
        if self.pending is not None:  # the motion is over: its answer, with the picture after it
            self._answer_motion(image, obs)
        while True:
            outcome = self.conversation.turn()
            if outcome[0] == "end":
                self.end = outcome[1]
                return self.robot.action(obs, None, self.gripper, terminate=True)
            _, name, args = outcome
            self.tools[name] += 1
            if name in ENDS:
                self.end = ENDS[name]
                return self.robot.action(obs, None, self.gripper, terminate=True)
            try:
                action = self._call(name, args, image, obs)
            except ValueError as e:
                self.conversation.answer([text_block(f"Not done: {e}.")], is_error=True)
                continue
            if action is not None:
                return action

    def _call(self, name: str, args: dict, image: np.ndarray, obs):
        """Does one tool call. A call that needs no simulator step is answered at once (None); a motion gives its
        first action, and its answer comes when it is over. ValueError: the call is not done, and why."""
        if name == "look":
            self.conversation.answer([picture_block(image)])
        elif name == "locate":
            x = int(number(args, "x", 0, image.shape[1] - 1))
            y = int(number(args, "y", 0, image.shape[0] - 1))
            point = self.robot.locate(obs, x, y)
            if point is None:
                raise ValueError(f"the camera has no depth at the pixel ({x}, {y})")
            self.conversation.answer([text_block("x = {:.3f}, y = {:.3f}, z = {:.3f} (meters)".format(*point))])
        elif name == "move_to":
            xyz = [number(args, k, -2.0, 2.0) for k in ("x", "y", "z")]
            turn = number(args, "turn", -MAX_TURN_DEG, MAX_TURN_DEG)
            problem, queue, goal = self.robot.plan_move(obs, xyz, turn, self.gripper == GRIPPER_CLOSED)
            if problem is not None:
                self.conversation.answer([text_block(f"The motion was not done: {problem}."), picture_block(image)],
                                         is_error=True)
                return None
            self.queue, self.goal, self.settled, self.pending = list(queue), goal, 0, "move_to"
            return self.step_action(obs)
        elif name == "gripper":
            if args.get("state") not in ("open", "close"):
                raise ValueError("`state` must be open or close")
            self.gripper = GRIPPER_OPEN if args["state"] == "open" else GRIPPER_CLOSED
            self.hold, self.pending = HOLD_STEPS - 1, "gripper"
            return self.robot.action(obs, None, self.gripper)
        else:
            raise ValueError(f"there is no tool `{name}`")
        return None

    def step_action(self, obs):
        """The next action of the motion that was just planned (a plan to where the hand already is has no pose)."""
        return self.robot.action(obs, self.queue.pop(0) if self.queue else self.goal, self.gripper)

    def _answer_motion(self, image: np.ndarray, obs) -> None:
        if self.pending == "move_to":
            reached = self.robot.distance(obs, self.goal) <= REACHED
            text = "The motion was done." if reached else \
                "The motion was not done: the gripper stopped before it reached the position."
            self.conversation.answer([text_block(text), picture_block(image)], is_error=not reached)
        else:
            state = "open" if self.gripper == GRIPPER_OPEN else "closed"
            self.conversation.answer([text_block(f"The gripper is {state}."), picture_block(image)])
        self.pending, self.goal = None, None

    def finish_episode(self, success: bool, info: dict) -> None:
        """The episode's line in frontier.AGENT_FILE and its conversation in transcripts/. An episode that the
        eval's step limit ended, while the model's part went on, ends with "step limit"."""
        c = self.conversation
        if c is None:
            return
        usage = {f: int(c.usage[f]) for f in frontier.USAGE_FIELDS}
        models = sorted({r["model"] for r in c.records if r.get("model")})
        record = {
            **{k: info.get(k) for k in ("env_name", "obj_episode_id", "layout_seed", "layout_per_episode", "steps")},
            "episode": self.key, "success": bool(success), "turns": c.turns, "end": self.end or "step limit",
            "refusal": c.refusal, "error": c.error, "tools": dict(self.tools), "usage": usage,
            "cost_usd": frontier.usage_cost(usage), "model_seconds": c.model_seconds, "models": models,
            "harness": HARNESS, "system_sha256": frontier.text_hash(c.system),
            "tools_sha256": frontier.text_hash(json.dumps(TOOLS, sort_keys=True)),
            "request": {"model": frontier.MODEL, "effort": frontier.EFFORT, "max_tokens": MAX_TOKENS,
                        "max_turns": self.max_turns},
        }
        save_transcript(os.path.join(self.out_dir, "transcripts"), self.key, c, record)
        with open(os.path.join(self.out_dir, frontier.AGENT_FILE), "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self.last_error = c.error if c.outage else None

    def visualize_epoch(self, predicted_raw_actions, images, save_path: str) -> None:
        pass


def save_transcript(folder: str, key: str, conversation: Conversation, record: dict) -> None:
    """An episode's conversation as it was sent: NAME.json (the system prompt, the messages, each turn's record)
    with every picture as a PNG file in NAME/, in the order of the conversation."""
    name = key.replace("/", "__")
    os.makedirs(os.path.join(folder, name), exist_ok=True)
    count = 0

    def stored(block):
        nonlocal count
        if not isinstance(block, dict):
            return block
        if block.get("type") == "image" and block.get("source", {}).get("type") == "base64":
            data = base64.standard_b64decode(block["source"]["data"])
            file = f"{name}/{count}.png"
            count += 1
            with open(os.path.join(folder, file), "wb") as f:
                f.write(data)
            return {"type": "image", "file": file}
        if isinstance(block.get("content"), list):
            return {**block, "content": [stored(b) for b in block["content"]]}
        return block

    messages = [{**m, "content": [stored(b) for b in m["content"]]} for m in conversation.messages]
    with open(os.path.join(folder, name + ".json"), "w", encoding="utf-8") as f:
        json.dump({"record": record, "system": conversation.system, "tools": TOOLS, "messages": messages,
                   "turns": conversation.records}, f)


# The simulator's side ---------------------------------------------------------------------------------------------


class Robot:
    """The simulator's side of the tools: the camera's clean depth and calibration, the planner's motion planning
    (mplib, seeded per episode) and the hand's commands, as the planner gives them."""

    def __init__(self, planner: str = "mplib", planner_seed: Optional[int] = 0, max_step_pos: float = 0.02,
                 max_step_rot: float = 0.15) -> None:
        from . import geometry, planner_client, planners  # the simulator's environment (OpenCV, SciPy, mplib)

        self.geometry, self.client, self.planners = geometry, planner_client, planners
        self.planner = planners.make_planner(planner)
        self.spheres = planners.RobotSpheres()
        self.planner_seed, self.max_step_pos, self.max_step_rot = planner_seed, max_step_pos, max_step_rot
        self.home = None  # the hand's orientation at the episode's first step
        self.robot_ids: List[int] = []

    @staticmethod
    def target(obs) -> np.ndarray:
        """The pose that the arm's controller holds now."""
        return np.asarray(obs["agent"]["controller"]["arm"]["target_pose"], dtype=np.float64)

    def start(self, obs, env, key: str) -> None:
        self.home = self.target(obs)[3:].copy()
        self.robot_ids = [link.id for link in env.unwrapped.agent.robot.get_links()]
        if self.planner_seed is not None:
            self.planner.seed(zlib.crc32(f"{self.planner_seed}/{key}".encode()) % 2147483647 + 1)

    def locate(self, obs, x: int, y: int) -> Optional[Tuple[float, float, float]]:
        """The base-frame position of the surface at a pixel, from the clean depth; None where there is no depth."""
        view = self.geometry.CameraView(obs)
        height, width = view.depth.shape
        if not (0 <= x < width and 0 <= y < height) or not 1e-3 < view.depth[y, x] < 2.0:
            return None
        return tuple(float(v) for v in view.pixel_points()[y, x])

    def orientation(self, turn_deg: float) -> np.ndarray:
        """The home orientation turned about the vertical axis (a quaternion, w first)."""
        from scipy.spatial.transform import Rotation

        home = Rotation.from_quat(self.home[[1, 2, 3, 0]])
        return (Rotation.from_euler("z", np.deg2rad(turn_deg)) * home).as_quat()[[3, 0, 1, 2]]

    def plan_move(self, obs, xyz, turn_deg: float, closed: bool):
        """(None, the hand's poses, the goal) of a move to a position, or (why there is none, [], the goal). A move of
        STRAIGHT or less goes in a straight line, as the planner's own short moves do; a longer one is planned by
        mplib around what the camera sees (without the arm and, with the gripper closed, without what it holds)."""
        target, goal = self.target(obs), np.r_[np.asarray(xyz, dtype=np.float64), self.orientation(turn_deg)]
        low, high = self.planners.WORKSPACE
        if not all(lo <= v <= hi for v, lo, hi in zip(goal[:3], low, high, strict=True)):
            return "the position is outside the robot's workspace", [], goal
        if np.linalg.norm(goal[:3] - target[:3]) <= STRAIGHT:
            return None, self.client.interpolate_poses(target, goal, self.max_step_pos, self.max_step_rot), goal
        qpos = np.asarray(obs["agent"]["qpos"], dtype=np.float64)
        view = self.geometry.CameraView(obs)
        view.robot_spheres = self.spheres(qpos)
        if closed:
            view.ignore_sphere = (self.client.ee_pose_from_obs(obs)[:3], 0.08)
        result = self.planner.plan(qpos, goal, view, self.robot_ids)
        if not result.success:
            return f"no path to the position was found ({result.status})", [], goal
        poses = [self.planner.fk(q) for q in result.path]
        poses[-1] = goal  # end on the goal, not on the planner's IK tolerance
        return None, self.client.resample_poses(poses, self.max_step_pos, self.max_step_rot), goal

    def distance(self, obs, goal: np.ndarray) -> float:
        return float(np.linalg.norm(self.client.ee_pose_from_obs(obs)[:3] - goal[:3]))

    def action(self, obs, desired: Optional[np.ndarray], gripper: float, terminate: bool = False):
        """The command that moves the controller's target to `desired` (None: it stays), as the planner's."""
        target = self.target(obs)
        world_vector, rot_axangle = self.geometry.ee_delta_action(
            target, target if desired is None else desired, self.client.ee_pose_from_obs(obs)[:3])
        raw_action = {"world_vector": world_vector, "rotation_delta": rot_axangle,
                      "open_gripper": np.array([1.0 if gripper > 0 else 0.0])}
        action = {"world_vector": world_vector, "rot_axangle": rot_axangle, "gripper": np.array([gripper]),
                  "terminate_episode": np.array([1.0 if terminate else 0.0])}
        return raw_action, action


# The model ----------------------------------------------------------------------------------------------------


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self.path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


class HttpRelay:
    """The model through the relay (experiments/agent_relay.py): one HTTP POST of the request's JSON for each turn.
    `address`: HOST:PORT, or unix:PATH for a Unix socket (an SSH tunnel's end on the GPU server, which only its
    owner can open). A relay that cannot be reached is tried again; once a request is sent, it is never sent again."""

    def __init__(self, address: str, timeout: float = 1800.0, tries: int = 5, wait: float = 15.0) -> None:
        self.address, self.timeout, self.tries, self.wait = address, timeout, tries, wait

    def connection(self) -> http.client.HTTPConnection:
        if self.address.startswith("unix:"):
            return _UnixConnection(self.address[len("unix:"):], self.timeout)
        host, _, port = self.address.rpartition(":")
        return http.client.HTTPConnection(host, int(port), timeout=self.timeout)

    def __call__(self, request: dict) -> dict:
        body = json.dumps(request).encode("utf-8")
        for attempt in range(self.tries):
            connection = self.connection()
            try:
                connection.connect()
            except OSError as e:
                connection.close()
                if attempt + 1 == self.tries:
                    raise ModelError(f"the relay at {self.address} cannot be reached: {e}") from None
                time.sleep(self.wait)
                continue
            try:
                connection.request("POST", "/", body=body, headers={"Content-Type": "application/json"})
                response = connection.getresponse()
                status, data = response.status, response.read()
            except (OSError, http.client.HTTPException) as e:
                raise ModelError(f"the relay gave no answer: {type(e).__name__}: {e}") from None
            finally:
                connection.close()
            try:
                answer = json.loads(data)
            except ValueError:
                raise ModelError(f"the relay's answer ({status}) is not JSON") from None
            if status != 200:
                raise ModelError(f"{status}: {answer.get('error') if isinstance(answer, dict) else answer}")
            return answer
        raise ModelError(f"the relay at {self.address} cannot be reached")


def scripted_reach() -> Callable[[dict], dict]:
    """A made-up model for pipeline checks (no request to anything): it looks, locates the picture's middle, moves
    to 10 cm above that point, closes and opens the gripper, goes 5 cm straight down and calls done."""
    def tool(n: int, name: str, **args) -> dict:
        return {"id": f"msg_scripted_{n}", "model": "scripted", "role": "assistant", "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": f"toolu_scripted_{n}", "name": name, "input": args}],
                "usage": dict.fromkeys(frontier.USAGE_FIELDS, 0)}

    def last_text(request: dict) -> str:
        blocks = request["messages"][-1]["content"][0].get("content") or []
        return " ".join(b.get("text", "") for b in blocks if isinstance(b, dict))

    state: Dict[str, object] = {}

    def ask(request: dict) -> dict:
        n = sum(m["role"] == "assistant" for m in request["messages"])
        if n == 0:
            state.clear()
            return tool(n, "look")
        if n == 1:
            width, height = (int(v) for v in re.search(r"(\d+) x (\d+) pixels", request["system"]).groups())
            return tool(n, "locate", x=width // 2, y=height // 2)
        if n == 2:
            found = re.search(r"x = (-?[\d.]+), y = (-?[\d.]+), z = (-?[\d.]+)", last_text(request))
            if not found:
                return tool(n, "give_up")
            state["xyz"] = [float(v) for v in found.groups()]
            x, y, z = state["xyz"]
            return tool(n, "move_to", x=x, y=y, z=z + 0.10, turn=0)
        if "xyz" not in state:
            return tool(n, "give_up")
        x, y, z = state["xyz"]
        steps = [("gripper", {"state": "close"}), ("gripper", {"state": "open"}),
                 ("move_to", {"x": x, "y": y, "z": z + 0.05, "turn": 0}), ("done", {})]
        name, args = steps[min(n - 3, len(steps) - 1)]
        return tool(n, name, **args)

    return ask
