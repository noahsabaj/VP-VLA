"""Demonstrations for the VLA, recorded during an eval (`--record-demos DIR`): each episode's camera pictures, the
command that the controller sent at each step and the hand's pose, in the VLA's own action space.

The VLA (model2simpler_interface_visual_prompt.py) predicts 7 numbers per step: the hand's movement (world_vector, in
metres), its turn as roll, pitch and yaw, and open (1) or closed (0). It turns the angles into an axis-angle turn
(transforms3d's euler2axangle: static x, y and z axes) and the open value into the gripper command (+1 open, -1
closed), and sends these to the simulator's controller. The planner sends the same controller the same kind of command
(planner_client.py, geometry.ee_delta_action). So every controller's command has exactly one VLA action:
label_from_action is the inverse of the VLA's conversion (action_from_label), and with that label the VLA would send
the same command.

Each episode becomes a folder DIR/EPISODE_KEY (the evaluator's key: TASK/LAYOUTS/EPISODE) with:
- frames.mp4: the camera picture of each step, before the step's command (the picture that the controller saw), at
  the eval's 5 steps per second;
- episode.npz: `actions` (the labels, steps x 7) and `states` (steps x 8: Bridge's state, see state_from_obs);
- meta.json, written last: the instruction of each step, the success, the steps and the episode's settings.
examples/SimplerEnv/train_files/planner_demos_to_lerobot.py makes a VP-VLA training set of these folders.
"""

import json
import os
from typing import Callable, Optional

import numpy as np

FPS = 5  # the evals' control frequency (--control-freq 5): one picture and one command per step


def euler_to_matrix(rpy) -> np.ndarray:
    """Roll, pitch and yaw about the static x, y and z axes (transforms3d's "sxyz", scipy's "xyz"): Rz Ry Rx."""
    r, p, y = (float(a) for a in rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def matrix_to_euler(m: np.ndarray) -> np.ndarray:
    """The inverse of euler_to_matrix (pitch in [-pi/2, pi/2])."""
    return np.array([np.arctan2(m[2, 1], m[2, 2]), np.arctan2(-m[2, 0], np.hypot(m[0, 0], m[1, 0])),
                     np.arctan2(m[1, 0], m[0, 0])])


def axangle_to_matrix(v) -> np.ndarray:
    """An axis-angle turn (axis times angle) as a rotation matrix (Rodrigues)."""
    v = np.asarray(v, float).reshape(3)
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def matrix_to_quat(m: np.ndarray) -> np.ndarray:
    """A rotation matrix as a unit quaternion (w, x, y, z) with w >= 0 (Shepperd's method)."""
    t = np.trace(m)
    if t > 0:
        s = 2 * np.sqrt(t + 1)
        q = [s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2 * np.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2 * np.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2 * np.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4]
    q = np.array(q)
    return -q if q[0] < 0 else q


def matrix_to_axangle(m: np.ndarray) -> np.ndarray:
    """A rotation matrix as an axis-angle turn (axis times angle, the angle in [0, pi])."""
    q = matrix_to_quat(m)
    n = np.linalg.norm(q[1:])
    if n < 1e-12:
        return np.zeros(3)
    return q[1:] / n * 2 * np.arctan2(n, q[0])


def label_from_action(action: dict) -> np.ndarray:
    """The VLA's 7 numbers (world_vector, roll, pitch, yaw, open) for a controller command: action["world_vector"],
    action["rot_axangle"] (an axis-angle turn) and action["gripper"] (+1 open, -1 closed)."""
    rpy = matrix_to_euler(axangle_to_matrix(action["rot_axangle"]))
    gripper = float(np.asarray(action["gripper"], float).reshape(-1)[0])
    return np.concatenate([np.asarray(action["world_vector"], float).reshape(3), rpy, [1.0 if gripper > 0 else 0.0]])


def action_from_label(label) -> dict:
    """The VLA's conversion of its 7 numbers into a controller command (model2simpler_interface_visual_prompt.py with
    policy_setup widowx_bridge and action scale 1)."""
    label = np.asarray(label, float)
    return {"world_vector": label[:3], "rot_axangle": matrix_to_axangle(euler_to_matrix(label[3:6])),
            "gripper": np.array([2.0 * (label[6] > 0.5) - 1.0])}


def pose_to_matrix(pose) -> np.ndarray:
    """A SAPIEN pose (x, y, z, qw, qx, qy, qz) as a 4 x 4 matrix."""
    pose = np.asarray(pose, float)
    w, x, y, z = pose[3:7] / np.linalg.norm(pose[3:7])
    m = np.eye(4)
    m[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    m[:3, 3] = pose[:3]
    return m


def state_from_obs(obs: dict, gripper_open: float) -> np.ndarray:
    """Bridge's 8 state numbers: the hand's position and roll, pitch and yaw in the robot's base frame (the
    controller's TCP, as planner_client.ee_pose_from_obs), 0, and the gripper (1 open, 0 closed: the last command).
    The training set needs them, but VP-VLA does not read the state (the data config's include_state is off)."""
    m = np.linalg.inv(pose_to_matrix(obs["agent"]["base_pose"])) @ pose_to_matrix(obs["extra"]["tcp_pose"])
    return np.concatenate([m[:3, 3], matrix_to_euler(m[:3, :3]), [0.0, gripper_open]])


def _write_mp4(path: str, frames, fps: int) -> None:
    import mediapy  # the simulator environment's video writer (as SimplerEnv's write_video)

    mediapy.write_video(path, frames, fps=fps)


class DemoRecorder:
    """Records each episode of an eval as a demonstration (see the module docstring). `write_video(path, frames,
    fps)` replaces mediapy's writer (tests)."""

    def __init__(self, root: str, write_video: Optional[Callable] = None) -> None:
        self.root = root
        self._write_video = write_video or _write_mp4
        self._episode: Optional[dict] = None

    def start(self, episode_key: str, **info) -> None:
        """A new episode; `info` (the task, the layouts, the episode) goes into its meta.json."""
        self._episode = {"key": episode_key, "info": info, "frames": [], "actions": [], "states": [],
                         "instructions": []}
        self._open = 1.0  # the WidowX starts with its gripper open

    def add(self, image, obs: dict, action: dict, instruction: str) -> None:
        """One step: the picture that the controller saw, the observation before the command, the command, and the
        instruction at this step."""
        e = self._episode
        e["frames"].append(np.asarray(image, np.uint8))
        e["states"].append(state_from_obs(obs, self._open))
        e["actions"].append(label_from_action(action))
        e["instructions"].append(instruction)
        self._open = float(e["actions"][-1][6])

    def finish(self, success: bool, **info) -> str:
        """Writes the episode's folder (meta.json last) and returns it."""
        e, self._episode = self._episode, None
        out = os.path.join(self.root, e["key"])
        os.makedirs(out, exist_ok=True)
        steps = len(e["actions"])
        np.savez(os.path.join(out, "episode.npz"), actions=np.array(e["actions"], np.float32).reshape(steps, 7),
                 states=np.array(e["states"], np.float32).reshape(steps, 8))
        height, width = e["frames"][0].shape[:2] if steps else (0, 0)
        if steps:
            self._write_video(os.path.join(out, "frames.mp4"), e["frames"], FPS)
        meta = {**e["info"], **info, "episode_key": e["key"], "success": bool(success), "steps": steps, "fps": FPS,
                "height": int(height), "width": int(width), "instructions": e["instructions"]}
        with open(os.path.join(out, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
        return out
