"""Planner stand-ins for ModelClientVP in SimplerEnv (KAN-22).

`PlannerClient` has the same reset()/step()/visualize_epoch() as ModelClientVP, but it
also reads the full observation (`obs`) and the simulator (`env`), which our copy of the
test loop passes in. Two modes:

* planner then VLA (`vla` given): plan from the camera to a pre-grasp spot above the
  object, move there, then hand every later step to the VLA.
* planner only (`vla=None`): same approach, then a scripted grasp, lift, planned move to
  above the place target, and release.

Targets come from the camera: SAM3 masks the object named in the instruction, and the
masked depth pixels give its 3D position. `target_source="sim"` reads them from the
simulator instead (debugging only). One JSON line per episode is written to `log_path`.
"""

import json
import math
import re
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Robocasa_tabletop" / "visual_prompt_utility"))

from .geometry import CameraView, DepthNoise, ee_delta_action, mat_to_pose, path_length, pose_to_mat
from .planners import make_planner

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

    The fingers close along the hand's y axis; the object's long side comes from PCA of its
    top-down points. Without points, keep the starting orientation.
    """
    if points_xy is None or len(points_xy) < 10:
        return home_quat
    home = Rotation.from_quat(home_quat[[1, 2, 3, 0]])
    closing = home.as_matrix()[:2, 1]
    centered = points_xy - points_xy.mean(axis=0)
    major = np.linalg.eigh(centered.T @ centered)[1][:, -1]
    want = np.arctan2(major[0], -major[1])  # perpendicular to the long side
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
        log_path: Optional[str] = None,
        depth_noise: str = "none",
        noise_seed: Optional[int] = None,
    ) -> None:
        self.planner = make_planner(planner)
        self.planner_name = planner
        self.vla = vla
        self.target_source = target_source
        self.pregrasp_height = pregrasp_height
        self.place_height = place_height
        self.max_step_pos = max_step_pos
        self.max_step_rot = max_step_rot
        self.settle_steps = settle_steps
        self.log_path = log_path
        self.noise = DepthNoise(depth_noise, noise_seed)
        self.sam3 = None
        if target_source == "camera":
            from sam3_client import SAM3Client

            self.sam3 = SAM3Client(host=sam3_host, port=sam3_port)
        self._clear("")

    # ------------------------------------------------------------------ episode

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
        self.noise.new_episode()
        self.stats = {"plans": [], "depth_noise": self.noise.level, "features": {}}
        self.ee_trace: List[np.ndarray] = []
        self.transit_contact = False
        self.place_goal = None

    def finish_episode(self, success: bool, info: dict) -> None:
        if self.log_path is None:
            return
        record = {"planner": self.planner_name, "mode": "planner_then_vla" if self.vla else "planner",
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
            self._start(image, obs, env, qpos, target)
        if self.phase == "vla":
            return self.vla.step(image, task_description)
        if self.phase == "done":
            return self._action(target, target, ee_pose, terminate=True)

        if self.phase.startswith("transit"):
            self.ee_trace.append(ee_pose[:3])
            if self._robot_touching(env):
                self.transit_contact = True

        if self.hold_steps > 0:  # gripper open/close in place
            self.hold_steps -= 1
            if self.hold_steps == 0:
                self._next_phase(env, obs, qpos, target)
            return self._action(target, target, ee_pose)

        if not self.queue:
            self._next_phase(env, obs, qpos, target)
            if self.phase == "vla":
                return self.vla.step(image, task_description)
            if self.phase == "done":
                return self._action(target, target, ee_pose, terminate=True)
            if self.hold_steps > 0 or not self.queue:
                return self._action(target, target, ee_pose)
        return self._action(target, self.queue.pop(0), ee_pose)

    # ------------------------------------------------------------------- phases

    def _start(self, image, obs, env, qpos, target) -> None:
        self.home_orientation = target[3:].copy()
        self.view = CameraView(obs, noise=self.noise)
        self.robot_ids = [link.id for link in env.unwrapped.agent.robot.get_links()]
        grasp = self._locate(image, obs, env, which="grasp")
        if grasp is None:
            self.stats["failure"] = "grasp object not found"
            self._give_up()
            return
        self.grasp = grasp
        # The VLA picks its own grasp after the hand-off, so only the planner-only setup turns the hand.
        orientation = self.home_orientation if self.vla is not None \
            else grasp_orientation(self.home_orientation, grasp.get("points_xy"))
        self.stats["features"]["grasp_turn_rad"] = rotation_angle(
            np.r_[np.zeros(3), self.home_orientation],
            np.r_[np.zeros(3), grasp_orientation(self.home_orientation, grasp.get("points_xy"))])
        self.stats["features"]["grasp_reach_m"] = float(np.linalg.norm(grasp["center"] - target[:2]))
        goal = np.concatenate([[grasp["center"][0], grasp["center"][1], grasp["top"] + self.pregrasp_height],
                               orientation])
        if not self._plan_to(qpos, goal, "pregrasp"):
            self._give_up()
            return
        self.phase = "transit_pregrasp"

    def _next_phase(self, env, obs, qpos, target) -> None:
        """Called when the current move or hold is finished."""
        phase = self.phase
        if phase.startswith("transit") and self._settle(target, obs):
            return
        if phase == "transit_pregrasp":
            self.stats["handoff_step"] = self.step_count
            self.stats["pregrasp_error"] = float(np.linalg.norm(ee_pose_from_obs(obs)[:3] - self.queue_goal[:3]))
            if self.vla is not None:
                self.phase = "vla"
                return
            grasp_z = max(0.5 * (self.grasp["top"] + self.grasp["bottom"]), self.grasp["bottom"] + 0.01)
            self._straight_to(target, np.r_[target[:2], grasp_z, target[3:]], "descend")
        elif phase == "descend":
            self.phase, self.gripper, self.hold_steps = "close", GRIPPER_CLOSED, 5
        elif phase == "close":
            self._straight_to(target, np.r_[target[:2], target[2] + 0.08, target[3:]], "lift")
        elif phase == "lift":
            place = self._locate(None, obs, env, which="place")
            if place is None:
                self.stats["failure"] = "place target not found"
                self.phase = "done"
                return
            goal = np.concatenate([[place["center"][0], place["center"][1],
                                    max(place["top"] + self.place_height, target[2])], self.home_orientation])
            self.view = CameraView(obs, noise=self.noise)
            self.view.ignore_sphere = (ee_pose_from_obs(obs)[:3], 0.08)  # the held object
            if not self._plan_to(qpos, goal, "place"):
                self.phase = "done"
                return
            self.phase = "transit_place"
        elif phase == "transit_place":
            self.phase, self.gripper, self.hold_steps = "open", GRIPPER_OPEN, 5
        elif phase == "open":
            self.phase = "done"

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
        self.phase = "vla" if self.vla is not None else "done"
        self.stats["handoff_step"] = self.step_count if self.vla is not None else None

    # --------------------------------------------------------------- perception

    def _locate(self, image, obs, env, which: str) -> Optional[dict]:
        """Center, top and bottom of the grasp object or place target, in the base frame."""
        truth = self._sim_object(obs, env, which)
        if self.target_source == "sim":
            return truth
        from sam3_client import extract_target_from_task

        name = extract_target_from_task(self.task_description) if which == "grasp" \
            else place_target_from_task(self.task_description)
        if not name:
            return None
        if image is None:
            image = obs["image"]["3rd_view_camera"]["rgb"]
        # SAM3 is often unsure of simulated objects (the spoon scores ~0.35, one eggplant 0.04),
        # so lower the bar until something turns up and keep the best-scoring mask.
        for threshold in SAM3_THRESHOLDS:
            seg = self.sam3.segment(image, name, threshold=threshold)
            masks, scores = seg.get("masks", []), np.asarray(seg.get("scores", []))
            if len(masks) > 0:
                break
        else:
            return None
        self.stats[f"{which}_sam3_score"] = float(scores.max())
        mask = np.asarray(masks[int(np.argmax(scores))])
        pts = CameraView(obs, noise=self.noise).object_points(mask)
        self._record_features(which, scores, mask, pts)
        if len(pts) < 5:
            return None
        found = {"center": np.median(pts[:, :2], axis=0), "top": float(np.percentile(pts[:, 2], 95)),
                 "bottom": float(np.percentile(pts[:, 2], 5)), "prompt": name, "points_xy": pts[:, :2]}
        if truth is not None:  # KAN-22 check: how far the camera estimate is from the real object
            center3 = np.r_[found["center"], 0.5 * (found["top"] + found["bottom"])]
            self.stats[f"{which}_error_xy"] = float(np.linalg.norm(found["center"] - truth["center"]))
            self.stats[f"{which}_error_3d"] = float(np.linalg.norm(center3 - truth["center3"]))
        return found

    def _record_features(self, which: str, scores, mask, pts) -> None:
        """What the adaptive selector sees about an object: all from the camera, none from the simulator."""
        f = self.stats["features"]
        f[f"{which}_sam3_score"] = float(np.max(scores))
        f[f"{which}_sam3_count"] = int(len(scores))
        f[f"{which}_mask_px"] = int(np.asarray(mask).astype(bool).sum())
        f[f"{which}_points"] = int(len(pts))
        if len(pts) >= 5:
            xy = pts[:, :2] - pts[:, :2].mean(axis=0)
            spread = np.sqrt(np.maximum(np.linalg.eigvalsh(xy.T @ xy / len(xy)), 0.0))
            f[f"{which}_height"] = float(np.percentile(pts[:, 2], 95) - np.percentile(pts[:, 2], 5))
            f[f"{which}_extent_long"], f[f"{which}_extent_short"] = float(4 * spread[1]), float(4 * spread[0])
            f[f"{which}_depth_spread"] = float(np.std(pts[:, 2]))

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
