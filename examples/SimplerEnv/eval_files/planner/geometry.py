"""Pose and camera geometry for the SimplerEnv planners.

Poses are 7-vectors [x, y, z, qw, qx, qy, qz] (SAPIEN convention). Everything the
planners see is expressed in the robot base frame.
"""

from typing import Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

CAMERA = "3rd_view_camera"


def pose_to_mat(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    mat = np.eye(4)
    mat[:3, 3] = pose[:3]
    mat[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    return mat


def mat_to_pose(mat: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(mat[:3, :3]).as_quat()
    return np.concatenate([mat[:3, 3], xyzw[[3, 0, 1, 2]]])


def ee_delta_action(target_pose: np.ndarray, desired_pose: np.ndarray, ee_pos: np.ndarray):
    """Hand movement that moves the controller's target from target_pose to desired_pose.

    Inverts ManiSkill2_real2sim's `ee_align2` target-delta rule (pd_ee_pose.py):
        new_target = T(ee_pos) * D * T(-ee_pos) * target
    where ee_pos is where the hand actually is. Returns (world_vector, rot_axangle).
    """
    r_target = pose_to_mat(target_pose)[:3, :3]
    r_desired = pose_to_mat(desired_pose)[:3, :3]
    r_delta = r_desired @ r_target.T
    world_vector = desired_pose[:3] - ee_pos - r_delta @ (target_pose[:3] - ee_pos)
    return world_vector, Rotation.from_matrix(r_delta).as_rotvec()


class DepthNoise:
    """A real depth camera's errors, so the planner doesn't get the simulator's perfect depth.

    * Per-pixel noise that grows with distance squared (RealSense-like: sigma = k * z^2).
    * Dropped pixels: random ones, plus holes along depth edges (object borders).
    * A camera pose error drawn once per episode (calibration error), in mm and degrees.
    """

    LEVELS = {
        "none": None,
        "realistic": dict(k=0.004, drop=0.02, edge_drop=0.5, edge_jump=0.02, pos_mm=3.0, rot_deg=0.3),
        "harsh": dict(k=0.012, drop=0.08, edge_drop=0.9, edge_jump=0.01, pos_mm=10.0, rot_deg=1.0),
    }

    def __init__(self, level: str = "none", seed: Optional[int] = None):
        self.level = level
        self.cfg = self.LEVELS[level]
        self.rng = np.random.default_rng(seed)
        self.pose_error = np.eye(4)

    @property
    def active(self) -> bool:
        return self.cfg is not None

    def new_episode(self) -> None:
        """Draw this episode's camera pose error (fixed for the whole episode, like a bad calibration)."""
        self.pose_error = np.eye(4)
        if not self.active:
            return
        axis = self.rng.normal(size=3)
        angle = np.deg2rad(self.cfg["rot_deg"]) * self.rng.normal()
        self.pose_error[:3, :3] = Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix()
        self.pose_error[:3, 3] = self.rng.normal(size=3) * self.cfg["pos_mm"] / 1000.0

    def apply(self, depth: np.ndarray) -> np.ndarray:
        if not self.active:
            return depth
        c = self.cfg
        valid = depth > 1e-3
        noisy = depth + self.rng.normal(size=depth.shape) * c["k"] * depth ** 2
        gy, gx = np.gradient(depth)
        edges = np.hypot(gx, gy) > c["edge_jump"]
        dropped = (self.rng.random(depth.shape) < c["drop"]) | (edges & (self.rng.random(depth.shape) < c["edge_drop"]))
        noisy[dropped | ~valid] = 0.0
        return noisy


class CameraView:
    """One observation's depth picture, turned into 3D points in the robot base frame."""

    def __init__(self, obs: dict, camera: str = CAMERA, noise: Optional[DepthNoise] = None):
        image = obs["image"][camera]
        param = obs["camera_param"][camera]
        self.rgb = image["rgb"]
        self.depth = image["depth"][..., 0].astype(np.float64)
        self.actor_seg = image["Segmentation"][..., 1]
        self.intrinsic = np.asarray(param["intrinsic_cv"], dtype=np.float64)
        world_from_cam = np.linalg.inv(np.asarray(param["extrinsic_cv"], dtype=np.float64))
        base_from_world = np.linalg.inv(pose_to_mat(obs["agent"]["base_pose"]))
        self.base_from_cam = base_from_world @ world_from_cam
        if noise is not None and noise.active:
            self.depth = noise.apply(self.depth)
            self.base_from_cam = self.base_from_cam @ noise.pose_error  # we believe the camera is where it isn't
        self.ignore_sphere = None  # (center, radius): e.g. the object held in the gripper

    def pixel_points(self) -> np.ndarray:
        """(H, W, 3) base-frame point for every pixel."""
        h, w = self.depth.shape
        v, u = np.mgrid[0:h, 0:w]
        z = self.depth
        fx, fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
        cx, cy = self.intrinsic[0, 2], self.intrinsic[1, 2]
        cam = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z, np.ones_like(z)], axis=-1)
        return (cam @ self.base_from_cam.T)[..., :3]

    def points(self, mask: Optional[np.ndarray] = None, max_depth: float = 2.0) -> np.ndarray:
        """Base-frame points for the pixels in mask (all valid pixels when mask is None)."""
        valid = (self.depth > 1e-3) & (self.depth < max_depth)
        if mask is not None:
            valid &= mask.astype(bool)
        return self.pixel_points()[valid]

    def obstacle_mask(self, robot_actor_ids) -> np.ndarray:
        """Pixels that count as obstacles: not the robot, not inside ignore_sphere."""
        mask = ~np.isin(self.actor_seg, np.asarray(list(robot_actor_ids)))
        if self.ignore_sphere is not None:
            center, radius = self.ignore_sphere
            mask &= np.linalg.norm(self.pixel_points() - np.asarray(center), axis=-1) > radius
        return mask

    def obstacle_cloud(self, robot_actor_ids, voxel: float = 0.01, workspace=None) -> np.ndarray:
        """Everything the camera sees except the robot, downsampled to one point per voxel."""
        pts = self.points(self.obstacle_mask(robot_actor_ids))
        if workspace is not None:
            lo, hi = np.asarray(workspace[0]), np.asarray(workspace[1])
            pts = pts[np.all((pts >= lo) & (pts <= hi), axis=1)]
        return voxel_downsample(pts, voxel)

    def object_points(self, mask: np.ndarray, erode_px: int = 2) -> np.ndarray:
        """Base-frame points of one segmented object, with mask edges and depth outliers dropped."""
        mask = mask.astype(np.uint8)
        if erode_px > 0 and mask.sum() > 50:
            mask = cv2.erode(mask, np.ones((2 * erode_px + 1,) * 2, np.uint8))
        pts = self.points(mask)
        if len(pts) < 5:
            return pts
        dist = np.linalg.norm(pts - np.median(pts, axis=0), axis=1)
        return pts[dist < max(0.05, 3 * np.median(dist))]


def voxel_downsample(points: np.ndarray, voxel: float) -> np.ndarray:
    if len(points) == 0 or voxel <= 0:
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx]


def path_length(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
