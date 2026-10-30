"""Planner stand-ins for ModelClientVP in SimplerEnv.

`PlannerClient` has the same reset()/step()/visualize_epoch() as ModelClientVP, but it
also reads the full observation (`obs`) and the simulator (`env`), which our copy of the
test loop passes in. Three modes:

* planner then VLA (`vla` given): plan from the camera to a pre-grasp spot above the
  object, move there, then hand every later step to the VLA.
* planner only (`vla=None`): same approach, then a scripted grasp, lift, planned move to
  above the place target, and release.
* selector (`vla` and `selector` given): after the planner's first look, the selector (the ASM)
  gives the whole try either to the planner (as in planner only) or to the VLA. With the VLM plan
  check (`plan_check_port`, plan_check.py), a "no" from the VLM gives it to the VLA first.

What to pick up and where to put it comes from the instruction: by default from one fixed rule (one
pick-and-place); with `task_source="vlm"` from the VLM plan (vlm_plan.py: the VLM lists the pick-and-place moves,
and the planner does them one after another; an instruction that is not pick-and-place goes to the VLA in the
selector mode).

Targets come from the camera: SAM3 masks the object named in the instruction, and the
masked depth pixels give its 3D position. `target_source="sim"` reads them from the
simulator instead (debugging only). One JSON line per episode is written to `log_path`.
"""

import json
import math
import re
import sys
import time
import zlib
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Robocasa_tabletop" / "visual_prompt_utility"))

from ..model_time import METER, time_requests
from . import plan_check, vlm_plan
from .geometry import (CameraView, DepthNoise, MissingDepth, corridor_points, depth_roughness, ee_delta_action,
                       mat_to_pose, path_length, pose_to_mat, short_width)
from .planners import FINGER_OPEN, WORKSPACE, RobotSpheres, make_planner
from .selector import FEATURES_VERSION, vlm_plan_gate

GRIPPER_OPEN, GRIPPER_CLOSED = 1.0, -1.0
PLACE_PREPOSITIONS = (" onto ", " on ", " into ", " in ")
SAM3_THRESHOLDS = (0.5, 0.3, 0.1, 0.03)


def place_target_from_task(task: str) -> Optional[str]:
    text = task.lower().strip()
    for prep in PLACE_PREPOSITIONS:
        if prep in text:
            place = text.split(prep, 1)[1].strip()
            return re.sub(r"^the ", "", place) or None
    return None


def ee_pose_from_obs(obs) -> np.ndarray:
    """Where the hand (ee_gripper_link, the controller's TCP) is, in the base frame."""
    base_from_world = np.linalg.inv(pose_to_mat(obs["agent"]["base_pose"]))
    return mat_to_pose(base_from_world @ pose_to_mat(obs["extra"]["tcp_pose"]))


def grasp_orientation(home_quat: np.ndarray, points_xy: Optional[np.ndarray]) -> np.ndarray:
    """Turn the hand about the vertical so the fingers close across the object's short side.

    The fingers close along the hand's y axis. For long objects the long side comes from PCA of
    the top-down points. For squarish ones (a cube) PCA's axis is arbitrary and can put the
    fingers on the corners, so use the smallest rectangle around the points to find a face.
    Without points, keep the starting orientation.
    """
    if points_xy is None or len(points_xy) < 10:
        return home_quat
    home = Rotation.from_quat(home_quat[[1, 2, 3, 0]])
    closing = home.as_matrix()[:2, 1]
    centered = points_xy - points_xy.mean(axis=0)
    evals, evecs = np.linalg.eigh(centered.T @ centered)
    if evals[1] > 1.7 * evals[0]:  # long side at least ~1.3x the short side
        major = evecs[:, -1]
        side = np.arctan2(major[1], major[0])
    else:
        (_, _), (w, h), angle = cv2.minAreaRect((points_xy * 1000.0).astype(np.float32))
        side = np.deg2rad(angle if w >= h else angle + 90.0)
    want = side + np.pi / 2  # close across the long side
    turn = (want - np.arctan2(closing[1], closing[0]) + np.pi / 2) % np.pi - np.pi / 2
    xyzw = (Rotation.from_euler("z", turn) * home).as_quat()
    return xyzw[[3, 0, 1, 2]]


def rotation_angle(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    r_a = pose_to_mat(pose_a)[:3, :3]
    r_b = pose_to_mat(pose_b)[:3, :3]
    return float(Rotation.from_matrix(r_a.T @ r_b).magnitude())


def interpolate_poses(a: np.ndarray, b: np.ndarray, max_pos: float, max_rot: float) -> List[np.ndarray]:
    """Poses from a (exclusive) to b (inclusive), no two neighbours further apart than the limits."""
    n = max(1, math.ceil(max(np.linalg.norm(b[:3] - a[:3]) / max_pos, rotation_angle(a, b) / max_rot)))
    rots = Rotation.from_quat(np.stack([a[[4, 5, 6, 3]], b[[4, 5, 6, 3]]]))
    slerp = Slerp([0.0, 1.0], rots)
    out = []
    for t in np.linspace(0.0, 1.0, n + 1)[1:]:
        xyzw = slerp([t]).as_quat()[0]
        out.append(np.concatenate([a[:3] + t * (b[:3] - a[:3]), xyzw[[3, 0, 1, 2]]]))
    return out


def resample_poses(poses: List[np.ndarray], max_pos: float, max_rot: float) -> List[np.ndarray]:
    """Fewest waypoints along the path such that each step stays within the limits."""
    dense = [poses[0]]
    for pose in poses[1:]:
        dense += interpolate_poses(dense[-1], pose, max_pos, max_rot)
    kept = [dense[0]]
    for i in range(1, len(dense)):
        too_far = (np.linalg.norm(dense[i][:3] - kept[-1][:3]) > max_pos
                   or rotation_angle(kept[-1], dense[i]) > max_rot)
        if too_far and dense[i - 1] is not kept[-1]:
            kept.append(dense[i - 1])
    if kept[-1] is not dense[-1]:
        kept.append(dense[-1])
    return kept[1:]  # the first pose is where the hand already is


class PlannerClient:
    def __init__(
        self,
        planner: str = "mplib",
        vla=None,
        sam3_host: str = "127.0.0.1",
        sam3_port: int = 10094,
        target_source: str = "camera",
        pregrasp_height: float = 0.06,
        place_height: float = 0.05,
        max_step_pos: float = 0.02,
        max_step_rot: float = 0.15,
        settle_steps: int = 3,
        min_grip: float = 0.0165,
        grasp_retries: int = 1,
        log_path: Optional[str] = None,
        depth_noise: str = "none",
        noise_seed: Optional[int] = None,
        rgb_noise: str = "none",
        robot_mask: str = "sim",
        selector=None,
        planner_seed: Optional[int] = None,
        plan_check_host: str = "127.0.0.1",
        plan_check_port: Optional[int] = None,
        task_source: str = "rule",
        vlm_plan_host: str = "127.0.0.1",
        vlm_plan_port: Optional[int] = None,
        vlm_plan_min_score: float = 0.0,
        plan_check_marks: Sequence[str] = ("tint",),
        plan_check_threshold: float = plan_check.THRESHOLD,
        missing_depth: str = "none",
    ) -> None:
        self.planner = make_planner(planner)
        self.planner_name = planner
        self.vla = vla
        # The ASM from selector.load_asm; it needs a VLA to give tries to.
        self.selector = selector
        if selector is not None:
            if vla is None:
                raise ValueError("the selector mode needs a VLA")
            if robot_mask != "joints":  # with "sim" the plan features would use the simulator's segmentation
                raise ValueError("the selector mode needs --robot-mask joints, so its features come from the camera only")
            trained = selector.trained_on
            # A selector saved before the RGB noise existed was trained without it.
            for name, value, old in (("depth_noise", depth_noise, None), ("rgb_noise", rgb_noise, "none"),
                                     ("robot_mask", robot_mask, None), ("task_source", task_source, "rule"),
                                     ("vlm_plan_min_score", vlm_plan_min_score, 0.0),
                                     ("missing_depth", missing_depth, "none")):
                if trained.get(name, old) not in (None, value):
                    raise ValueError(f"the selector was trained with {name}={trained.get(name, old)}, but this run "
                                     f"uses {value}")
        self.mode = "selector" if selector is not None else "planner_then_vla" if vla is not None else "planner"
        # Planner then VLA: the VLA takes over above the object. In the selector mode the planner
        # works exactly as in planner only, because that is what the selector was trained on.
        self.hand_off = self.mode == "planner_then_vla"
        self.target_source = target_source
        self.pregrasp_height = pregrasp_height
        self.place_height = place_height
        self.max_step_pos = max_step_pos
        self.max_step_rot = max_step_rot
        self.settle_steps = settle_steps
        self.min_grip = min_grip  # finger joint (m); fully closed on nothing reads 0.015
        self.grasp_retries = grasp_retries
        self.log_path = log_path
        self.noise = DepthNoise(depth_noise, noise_seed)
        # Surfaces the depth camera cannot measure (geometry.MissingDepth): with "target", the place target in about
        # half of the episodes.
        self.missing_depth = MissingDepth(missing_depth, noise_seed)
        # The evaluator puts the RGB noise into the picture (camera_noise.py); the planner only logs its level.
        self.rgb_noise = rgb_noise
        # The planner's random steps (MPlib: IK restarts, RRT) are seeded per episode like the depth noise, so a
        # try plans the same in every run, whatever ran before it in the process. None leaves them unseeded.
        if planner_seed is not None and not hasattr(self.planner, "seed"):
            raise ValueError(f"--planner-seed needs a planner that can be seeded (mplib), not {planner}")
        self.planner_seed = planner_seed
        self.episode_key: Optional[str] = None
        # How the robot is cut out of the obstacle map: the simulator's segmentation ("sim"),
        # or spheres placed from the joint angles ("joints"), which a real robot can do too.
        self.robot_mask = robot_mask
        self.robot_spheres = RobotSpheres() if robot_mask == "joints" else None
        self.sam3 = None
        if target_source == "camera":
            from sam3_client import SAM3Client

            # Every server request is timed for the episode's cost record (model_time.py).
            self.sam3 = time_requests(SAM3Client(host=sam3_host, port=sam3_port), "sam")
        # The SAM server's model (SAM3Model or SAM31Model), from its greeting: a run setting of every record.
        self.sam_model = (getattr(self.sam3, "_server_metadata", None) or {}).get("model") if self.sam3 else None
        # Every selector saved before the records had sam_model was trained on SAM3 (SAM3Model) runs.
        if selector is not None and self.sam_model is not None \
                and selector.trained_on.get("sam_model", "SAM3Model") != self.sam_model:
            raise ValueError(f"the selector was trained with {selector.trained_on.get('sam_model', 'SAM3Model')}, "
                             f"but this run's SAM server is {self.sam_model}")
        # The VLM plan check (plan_check.py): on the first frame, does the planner's one pick-and-place finish the
        # instruction? In the planner-only mode it only adds the feature vlm_plan_yes; the planner works as before.
        self.plan_check = None
        if plan_check_port is not None:
            if target_source != "camera":
                raise ValueError("the plan check needs --target-source camera: it shows the VLM SAM3's masks")
            from ..vlm_client_simpler import VLMClientSimpler

            self.plan_check = time_requests(VLMClientSimpler(host=plan_check_host, port=plan_check_port), "vlm")
        # How the object is marked (plan_check.MARKS), each mark its own question and feature; the selector mode's
        # gate uses the first one, with this threshold.
        self.plan_check_marks = list(plan_check_marks)
        for mark in self.plan_check_marks:
            plan_check.feature(mark)
        if not self.plan_check_marks or len(set(self.plan_check_marks)) < len(self.plan_check_marks):
            raise ValueError(f"plan_check_marks must name each mark once: {self.plan_check_marks}")
        self.plan_check_threshold = plan_check_threshold
        # Where the object and target come from: the fixed rule, or the VLM plan (vlm_plan.py).
        if task_source not in ("rule", "vlm"):
            raise ValueError(f"task_source must be rule or vlm, not {task_source}")
        self.task_source = task_source
        self.vlm_planner = None
        if task_source == "vlm":
            if target_source != "camera" or vlm_plan_port is None:
                raise ValueError("the VLM plan needs --target-source camera and --vlm-plan-port")
            if plan_check_port is not None:
                raise ValueError("the plan check asks about the fixed rule's one pick-and-place: not with the VLM plan")
            from ..vlm_client_simpler import VLMClientSimpler

            self.vlm_planner = time_requests(VLMClientSimpler(host=vlm_plan_host, port=vlm_plan_port), "vlm")
        # The VLM plan's SAM score check: a name whose chosen mask scores below this is not found (0 = no check).
        if vlm_plan_min_score and task_source != "vlm":
            raise ValueError("vlm_plan_min_score is for the VLM plan (task_source vlm)")
        self.vlm_plan_min_score = vlm_plan_min_score
        self._clear("")

    # ------------------------------------------------------------------ episode

    def set_episode(self, key: str) -> None:
        """Called by planner_evaluator before reset(): this episode's depth noise and planner seed come from `key`."""
        self.episode_key = key

    def reset(self, task_description: str) -> None:
        self._clear(task_description)
        if self.vla is not None:
            self.vla.reset(task_description)

    def _clear(self, task_description: str) -> None:
        self.task_description = task_description
        self.phase = "start"
        self.queue: List[np.ndarray] = []
        self.gripper = GRIPPER_OPEN
        self.hold_steps = 0
        self.step_count = 0
        self.noise.new_episode(self.episode_key)
        self.target_depth_missing = self.missing_depth.new_episode(self.episode_key)
        self.no_depth_ids: List[int] = []  # the target's actor when it gives no depth (set in _start)
        if self.planner_seed is not None:
            self.planner.seed(zlib.crc32(f"{self.planner_seed}/{self.episode_key}".encode()) % 2147483647 + 1)
        # The run settings go into every record, so the selector can refuse to mix runs.
        self.stats = {"plans": [], "depth_noise": self.noise.level, "rgb_noise": self.rgb_noise,
                      "noise_seed": self.noise.seed, "planner_seed": self.planner_seed, "robot_mask": self.robot_mask,
                      "features_version": FEATURES_VERSION, "features": {}, "plan_check": self.plan_check is not None,
                      "task_source": self.task_source, "vlm_plan_min_score": self.vlm_plan_min_score,
                      "sam_model": self.sam_model, "missing_depth": self.missing_depth.mode,
                      "target_depth_missing": self.target_depth_missing}
        if self.plan_check is not None:
            self.stats.update(plan_check_marks=self.plan_check_marks, plan_check_threshold=self.plan_check_threshold)
        # The VLM plan's pick-and-place moves (vlm_plan.parse) and the one being done; None with the fixed rule.
        self.steps = None
        self.step_index = 0
        self.ee_trace: List[np.ndarray] = []
        self.transit_contact = False
        self.place_goal = None
        self.retries_left = self.grasp_retries
        self.first_look = True  # the selector's features come from the first frame only

    def finish_episode(self, success: bool, info: dict) -> None:
        if self.log_path is None:
            return
        record = {"planner": self.planner_name, "mode": self.mode,
                  "task": self.task_description, "success": bool(success), **info, **self.stats,
                  "ee_path_length_planner": path_length(np.array(self.ee_trace)) if self.ee_trace else 0.0,
                  "transit_contact": self.transit_contact}
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, default=float) + "\n")

    def visualize_epoch(self, predicted_raw_actions, images, save_path: str) -> None:
        if self.vla is not None:
            self.vla.visualize_epoch(predicted_raw_actions, images, save_path)

    # --------------------------------------------------------------------- step

    def step(self, image: np.ndarray, task_description: Optional[str] = None, obs=None, env=None, **kwargs):
        if obs is None or env is None:
            raise ValueError("PlannerClient needs obs and env; run it with planner_evaluator")
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)
        self.step_count += 1
        if self.phase == "vla":
            return self.vla.step(image, task_description)

        qpos = np.asarray(obs["agent"]["qpos"], dtype=np.float64)
        target = np.asarray(obs["agent"]["controller"]["arm"]["target_pose"], dtype=np.float64)
        ee_pose = ee_pose_from_obs(obs)

        if self.phase == "start":
            t0, c0 = time.perf_counter(), time.process_time()
            self._start(image, obs, env, qpos, target)
            if self.selector is not None:
                self._choose()
            # What the episode cost up to here (model_time.py): the first look, which the ASM needs in every episode,
            # and in the selector mode the VLA's reset before it (the model requests and, in the energy runs, the
            # GPU's energy since the episode's start; None in other runs).
            self.stats["first_look_seconds"] = time.perf_counter() - t0
            self.stats["first_look_cpu_seconds"] = time.process_time() - c0
            self.stats["first_look_model"] = METER.snapshot()
            self.stats["first_look_gpu_joules"] = METER.gpu_joules()
        if (out := self._end_action(image, task_description, target, ee_pose)) is not None:
            return out

        if self.phase.startswith("transit"):
            self.ee_trace.append(ee_pose[:3])
            # The held object is known only for the first move (the task's first goal), so the check stops there.
            if self.step_index == 0 and self._robot_touching(env):
                self.transit_contact = True

        if self.hold_steps > 0:  # gripper open/close in place
            self.hold_steps -= 1
            if self.hold_steps == 0:
                self._next_phase(env, obs, qpos, target)
            return self._action(target, target, ee_pose)

        if not self.queue:
            self._next_phase(env, obs, qpos, target)
            if (out := self._end_action(image, task_description, target, ee_pose)) is not None:
                return out
            if self.hold_steps > 0 or not self.queue:
                return self._action(target, target, ee_pose)
        return self._action(target, self.queue.pop(0), ee_pose)

    def _end_action(self, image, task_description, target, ee_pose):
        """Once the planner's part is over: the VLA's action (phase "vla") or a stop (phase "done"); else None."""
        if self.phase == "vla":
            return self.vla.step(image, task_description)
        if self.phase == "done":
            return self._action(target, target, ee_pose, terminate=True)
        return None

    # ------------------------------------------------------------------- phases

    def _start(self, image, obs, env, qpos, target) -> None:
        self.home_orientation = target[3:].copy()
        self.robot_ids = [link.id for link in env.unwrapped.agent.robot.get_links()]
        if self.target_depth_missing:
            self.no_depth_ids = [env.unwrapped.episode_target_obj.id]
        self._approach(image, obs, env, qpos, target)

    def _choose(self) -> None:
        """Selector mode: from the first look's features and first plan, give the whole try to the
        planner or to the VLA. The robot has not moved yet, so the VLA starts as in a VLA-only run."""
        if self.episode_key is None:
            raise ValueError("the selector mode needs the episode key (planner_evaluator sets it): its task")
        p_yes = self.stats["features"].get(plan_check.feature(self.plan_check_marks[0])) \
            if self.plan_check is not None else None
        if (gate := vlm_plan_gate(self.stats)) is not None:  # the VLM plan found no pick-and-place moves
            decision = gate
        elif p_yes is not None and plan_check.gate(p_yes, self.plan_check_threshold) == "vla":
            decision = {"choice": "vla", "reason": "plan check"}
        else:
            decision = self.selector.decide(self.stats, self.episode_key.split("/")[0])
        self.stats.update({f"selector_{k}": v for k, v in decision.items()})
        if decision["choice"] == "vla":
            self.phase, self.queue = "vla", []

    def _approach(self, image, obs, env, qpos, target) -> None:
        """Find the object and plan to just above it."""
        first, self.first_look = self.first_look, False  # a grasp retry comes back here: no new features
        if first and self.vlm_planner is not None and not self._vlm_plan(image):
            self._give_up()
            return
        self.view = self._obstacle_view(obs, qpos)
        # One frame, one noisy depth picture: the object estimate and the obstacle map share it.
        grasp = self._locate(image, obs, env, which="grasp", record=first, view=self.view)
        if first:  # the place target too, for the selector only: placing finds it again after the lift
            place = self._locate(image, obs, env, which="place", record=True, view=self.view)
            if self.plan_check is not None:
                self._check_plan(image, grasp, place)
            if any(s["rejected"] for s in self.stats.get("vlm_plan_sam", [])):
                self.stats["failure"] = "VLM plan: SAM score below the minimum"
                self._give_up()
                return
        if grasp is None:
            self.stats["failure"] = "grasp object not found"
            self._give_up()
            return
        self.grasp = grasp
        # The VLA picks its own grasp after the hand-off, so only the planner-only setup turns the hand.
        orientation = self.home_orientation if self.hand_off \
            else grasp_orientation(self.home_orientation, grasp.get("points_xy"))
        goal = np.concatenate([[grasp["center"][0], grasp["center"][1], grasp["top"] + self.pregrasp_height],
                               orientation])
        if first:
            f = self.stats["features"]
            f["grasp_turn_rad"] = rotation_angle(
                np.r_[np.zeros(3), self.home_orientation],
                np.r_[np.zeros(3), grasp_orientation(self.home_orientation, grasp.get("points_xy"))])
            f["grasp_reach_m"] = float(np.linalg.norm(grasp["center"] - target[:2]))
            cloud = self.view.obstacle_cloud(self.robot_ids, voxel=0.01, workspace=WORKSPACE)
            f["pregrasp_corridor_points"] = corridor_points(cloud, ee_pose_from_obs(obs)[:3], goal[:3])
        if not self._plan_to(qpos, goal, "pregrasp"):
            self._give_up()
            return
        self.phase = "transit_pregrasp"

    def _vlm_plan(self, image) -> bool:
        """The VLM plan (vlm_plan.py) from the first frame: False when the VLM finds no pick-and-place moves."""
        answer = self.vlm_planner.generate([image], vlm_plan.prompt(self.task_description))
        plan = vlm_plan.parse(answer["text"], image.shape[1], image.shape[0])
        self.stats.update({"vlm_plan": plan, "vlm_plan_text": answer["text"], "vlm_plan_feasible": plan["feasible"],
                           "steps_done": 0, "vlm_plan_sam": []})
        if not plan["feasible"]:
            self.stats["failure"] = "VLM plan: not pick-and-place" + (f" ({plan['error']})" if plan["error"] else "")
            return False
        self.steps = plan["steps"]
        return True

    def _check_plan(self, image, grasp, place) -> None:
        """The VLM plan check (plan_check.py) on the first look, once per mark: its P(yes) that the one pick-and-place
        finishes the instruction is the mark's feature (plan_check.feature: vlm_plan_yes for the tint). With no object
        or no target found there is no plan to show: 0."""
        f = self.stats["features"]
        for mark in self.plan_check_marks:
            name, mass = plan_check.feature(mark), "plan_check_answer_mass" + ("" if mark == "tint" else f"_{mark}")
            if grasp is None or place is None:
                f[name], self.stats[mass] = 0.0, None
                continue
            marked = plan_check.mark_plan(image, grasp["mask"], place["mask"], mark=mark)
            answer = self.plan_check.yes_probability([image, marked], plan_check.question(self.task_description, mark))
            f[name], self.stats[mass] = float(answer["p_yes"]), float(answer["answer_mass"])
            if self.log_path is not None and self.episode_key is not None:  # the marked picture, to check by eye
                stem = self.episode_key.replace("/", "_") + ("" if mark == "tint" else f"_{mark}")
                path = Path(self.log_path).parent / "plan_check" / f"{stem}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path), np.ascontiguousarray(marked[..., ::-1]))

    def _next_phase(self, env, obs, qpos, target) -> None:
        """Called when the current move or hold is finished."""
        phase = self.phase
        if phase.startswith("transit") and self._settle(target, obs):
            return
        if phase == "transit_pregrasp":
            self.stats["handoff_step"] = self.step_count
            self.stats["pregrasp_error"] = float(np.linalg.norm(ee_pose_from_obs(obs)[:3] - self.queue_goal[:3]))
            if self.hand_off:
                self.phase = "vla"
                return
            grasp_z = max(0.5 * (self.grasp["top"] + self.grasp["bottom"]), self.grasp["bottom"] + 0.01)
            self.hold_offset = grasp_z - self.grasp["bottom"]  # hand height above the held object's bottom
            self._straight_to(target, np.r_[target[:2], grasp_z, target[3:]], "descend")
        elif phase == "descend":
            self.phase, self.gripper, self.hold_steps = "close", GRIPPER_CLOSED, 5
        elif phase == "close":
            self._straight_to(target, np.r_[target[:2], target[2] + 0.08, target[3:]], "lift")
        elif phase == "lift":
            grip = float(np.mean(qpos[6:8]))
            self.stats.setdefault("grip_after_lift", []).append(grip)
            if grip < self.min_grip:  # fingers closed on nothing
                if self.retries_left > 0:
                    self.retries_left -= 1
                    self.stats["grasp_retries"] = self.stats.get("grasp_retries", 0) + 1
                    self.phase, self.gripper, self.hold_steps = "reopen", GRIPPER_OPEN, 5
                    return
                self.stats["failure"] = "grasp missed"
                self.phase = "done"
                return
            # One frame, one noisy depth picture: the target estimate and the obstacle map share it.
            self.view = self._obstacle_view(obs, qpos)
            place = self._locate(None, obs, env, which="place", view=self.view)
            if place is None:
                self.stats["failure"] = "place target not found"
                self.phase = "done"
                return
            self.place_top = place["top"]
            goal = np.concatenate([[place["center"][0], place["center"][1],
                                    max(place["top"] + self.place_height, target[2])], self.home_orientation])
            self.view.ignore_sphere = (ee_pose_from_obs(obs)[:3], 0.08)  # the held object
            if not self._plan_to(qpos, goal, "place"):
                self.phase = "done"
                return
            self.phase = "transit_place"
        elif phase == "reopen":
            self._approach(None, obs, env, qpos, target)
        elif phase == "transit_place":
            # Lower until the held object is 1 cm above the target, so it isn't dropped from height.
            release_z = self.place_top + self.hold_offset + 0.01
            if release_z < target[2] - 0.005:
                self._straight_to(target, np.r_[target[:2], release_z, target[3:]], "lower")
            else:
                self.phase, self.gripper, self.hold_steps = "open", GRIPPER_OPEN, 5
        elif phase == "lower":
            self.phase, self.gripper, self.hold_steps = "open", GRIPPER_OPEN, 5
        elif phase == "open":
            if self.steps is None:
                self.phase = "done"
                return
            self.stats["steps_done"] += 1
            if self.step_index + 1 == len(self.steps):
                self.phase = "done"
                return
            # The VLM plan's next move: rise clear of the object just placed, then look and plan again.
            self.step_index += 1
            self.retries_left = self.grasp_retries
            self._straight_to(target, np.r_[target[:2], target[2] + 0.06, target[3:]], "retreat")
        elif phase == "retreat":
            self._approach(None, obs, env, qpos, target)

    def _obstacle_view(self, obs, qpos) -> CameraView:
        view = CameraView(obs, noise=self.noise, no_depth_ids=self.no_depth_ids)
        if self.robot_spheres is not None:
            view.robot_spheres = self.robot_spheres(qpos)
        return view

    def _plan_to(self, qpos, goal, label: str) -> bool:
        result = self.planner.plan(qpos, goal, self.view, self.robot_ids)
        record = {"label": label, "success": result.success, "time": result.plan_time, "status": result.status}
        self.stats["plans"].append(record)
        if not result.success:
            self.stats["failure"] = f"{label} plan failed: {result.status}"
            return False
        poses = [self.planner.fk(q) for q in result.path]
        record["joint_path_length"] = float(np.abs(np.diff(result.path, axis=0)).sum())
        record["ee_path_length"] = path_length(np.array([p[:3] for p in poses]))
        record["ee_straight_length"] = float(np.linalg.norm(np.asarray(goal[:3]) - poses[0][:3]))
        poses[-1] = goal  # end exactly on the goal, not on the planner's IK tolerance
        self.queue = resample_poses(poses, self.max_step_pos, self.max_step_rot)
        self.queue_goal = goal
        self.settled = 0
        return True

    def _settle(self, target, obs) -> bool:
        """Hold on the last waypoint for a few steps until the hand actually gets there."""
        if self.settled < self.settle_steps and np.linalg.norm(ee_pose_from_obs(obs)[:3] - target[:3]) > 0.01:
            self.settled += 1
            self.queue = [target.copy()]
            return True
        return False

    def _straight_to(self, target, goal, phase: str) -> None:
        self.queue = interpolate_poses(target, goal, self.max_step_pos, self.max_step_rot)
        self.phase = phase

    def _give_up(self) -> None:
        self.phase = "vla" if self.hand_off else "done"
        self.stats["handoff_step"] = self.step_count if self.hand_off else None

    # --------------------------------------------------------------- perception

    def _locate(self, image, obs, env, which: str, record: bool = False,
                view: Optional[CameraView] = None) -> Optional[dict]:
        """Center, top and bottom of the grasp object or place target, in the base frame.

        `record` writes the selector's features for this object. `view` reuses a depth picture
        already taken (and noised) for this frame instead of taking a new one.
        """
        # The simulator's object and target are those of the task's first goal: a check for the first move only.
        truth = self._sim_object(obs, env, which) if self.step_index == 0 else None
        if self.target_source == "sim":
            return truth
        from sam3_client import extract_target_from_task

        point = None
        if self.steps is not None:  # the VLM plan's current move: a name for SAM3 and a point to pick the mask
            step = self.steps[self.step_index]
            name, point = (step["object"], step["object_point"]) if which == "grasp" \
                else (step["target"], step["target_point"])
        else:
            name = extract_target_from_task(self.task_description) if which == "grasp" \
                else place_target_from_task(self.task_description)
        if not name:
            return None
        if image is None:
            image = obs["image"]["3rd_view_camera"]["rgb"]
        # SAM3 is often unsure of simulated objects (the spoon scores ~0.35, one eggplant 0.04),
        # so lower the bar until something turns up and keep the best-scoring mask. With a point (the VLM plan),
        # lower it further until a mask covers the point, if one ever does.
        found_any = None
        for threshold in SAM3_THRESHOLDS:
            seg = self.sam3.segment(image, name, threshold=threshold)
            masks, scores = seg.get("masks", []), np.asarray(seg.get("scores", []))
            if len(masks) == 0:
                continue
            found_any = found_any or (masks, scores)
            if point is None or any(vlm_plan.contains(m, point) for m in masks):
                break
        else:
            if found_any is None:
                return None
            masks, scores = found_any
        chosen = vlm_plan.pick_mask(masks, scores, point)
        score = float(scores[chosen])
        self.stats[f"{which}_sam3_score"] = score
        mask = np.asarray(masks[chosen])
        view = view if view is not None else CameraView(obs, noise=self.noise, no_depth_ids=self.no_depth_ids)
        pts = view.object_points(mask)
        if record:
            self._record_features(which, scores, mask, pts, view)
        if self.steps is not None:  # the VLM plan: each name's SAM score, and the score check
            rejected = score < self.vlm_plan_min_score
            covers = point is not None and vlm_plan.contains(mask, point)
            self.stats["vlm_plan_sam"].append({"step": self.step_index, "which": which, "name": name, "score": score,
                                               "covers_point": covers, "rejected": rejected})
            if rejected:
                return None
        if len(pts) < 5:
            return None
        found = {"center": np.median(pts[:, :2], axis=0), "top": float(np.percentile(pts[:, 2], 95)),
                 "bottom": float(np.percentile(pts[:, 2], 5)), "prompt": name, "points_xy": pts[:, :2], "mask": mask}
        if truth is not None:  # check: how far the camera estimate is from the real object
            center3 = np.r_[found["center"], 0.5 * (found["top"] + found["bottom"])]
            self.stats[f"{which}_error_xy"] = float(np.linalg.norm(found["center"] - truth["center"]))
            self.stats[f"{which}_error_3d"] = float(np.linalg.norm(center3 - truth["center3"]))
        return found

    def _record_features(self, which: str, scores, mask, pts, view: CameraView) -> None:
        """What the adaptive selector sees about an object: all from the camera, none from the simulator."""
        f = self.stats["features"]
        scores = np.sort(np.asarray(scores, dtype=float))
        m = np.asarray(mask).astype(bool)
        f[f"{which}_sam3_score"] = float(scores[-1])
        f[f"{which}_sam3_count"] = len(scores)
        # second-best over best SAM3 score: near 1 when two objects look alike
        f[f"{which}_sam3_ambiguity"] = float(scores[-2] / max(scores[-1], 1e-6)) if len(scores) > 1 else 0.0
        f[f"{which}_mask_px"] = int(m.sum())
        f[f"{which}_points"] = len(pts)
        f[f"{which}_valid_depth_frac"] = float((m & (view.depth > 1e-3) & (view.depth < 2.0)).sum() / max(m.sum(), 1))
        roughness = depth_roughness(view.depth, m)
        f[f"{which}_depth_roughness"] = roughness
        if len(pts) >= 5:
            xy = pts[:, :2] - pts[:, :2].mean(axis=0)
            spread = np.sqrt(np.maximum(np.linalg.eigvalsh(xy.T @ xy / len(xy)), 0.0))
            f[f"{which}_height"] = float(np.percentile(pts[:, 2], 95) - np.percentile(pts[:, 2], 5))
            f[f"{which}_extent_long"], f[f"{which}_extent_short"] = float(4 * spread[1]), float(4 * spread[0])
            f[f"{which}_depth_spread"] = float(np.std(pts[:, 2]))
            width = short_width(pts[:, :2])
            f[f"{which}_width"] = width
            f[f"{which}_depth_roughness_rel"] = roughness / max(width, 0.005)  # noise against object size
            if which == "grasp":  # room left in the open hand; each finger opens from 0.015 to FINGER_OPEN
                span = 2 * (FINGER_OPEN - 0.015)
                f["grasp_grip_margin"] = (span - width) / span

    @staticmethod
    def _sim_object(obs, env, which: str) -> Optional[dict]:
        u = env.unwrapped
        actor = u.episode_source_obj if which == "grasp" else u.episode_target_obj
        if actor is None:
            return None
        base_from_world = np.linalg.inv(pose_to_mat(obs["agent"]["base_pose"]))
        world = pose_to_mat(np.concatenate([actor.pose.p, actor.pose.q]))
        center = (base_from_world @ world)[:3, 3]
        half = 0.5 * float(np.asarray(u.episode_source_obj_bbox_world if which == "grasp"
                                      else u.episode_target_obj_bbox_world)[2])
        return {"center": center[:2], "center3": center, "top": center[2] + half, "bottom": center[2] - half}

    def _robot_touching(self, env) -> bool:
        """Is the robot touching anything (table, objects) other than what it's carrying?"""
        robot = set(self.robot_ids)
        held = {env.unwrapped.episode_source_obj.id} if self.phase == "transit_place" else set()
        for contact in env.unwrapped._scene.get_contacts():
            ids = {contact.actor0.id, contact.actor1.id}
            if ids & held:
                continue
            if len(ids & robot) == 1 and any(np.linalg.norm(p.impulse) > 1e-6 for p in contact.points):
                return True
        return False

    # ------------------------------------------------------------------- output

    def _action(self, target, desired, ee_pose, terminate: bool = False):
        world_vector, rot_axangle = ee_delta_action(target, desired, ee_pose[:3])
        gripper = np.array([self.gripper])
        raw_action = {"world_vector": world_vector, "rotation_delta": rot_axangle,
                      "open_gripper": np.array([1.0 if self.gripper > 0 else 0.0])}
        action = {"world_vector": world_vector, "rot_axangle": rot_axangle, "gripper": gripper,
                  "terminate_episode": np.array([1.0 if terminate else 0.0])}
        return raw_action, action
