"""MPlib and cuRobo behind one interface, for the SimplerEnv WidowX.

Both plan in the robot base frame for the 6 arm joints (fingers held open), and both
see obstacles only through the camera: MPlib gets a point cloud, cuRobo integrates the
depth picture into its own map. `plan()` returns joint waypoints, `fk()` the hand pose.
"""

import os
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .geometry import CameraView, mat_to_pose

EE_LINK = "ee_gripper_link"
ARM_DOF = 6
FINGER_OPEN = 0.037
# Base-frame box around the table the planners care about (robot faces +x, table top is near z=0).
WORKSPACE = ((-0.15, -0.45, -0.05), (0.75, 0.45, 0.55))


def widowx_asset_dir() -> Path:
    """A writable copy of SimplerEnv's WidowX description (MPlib writes an SRDF next to the URDF)."""
    src = Path(os.environ["SIMPLERENV_PATH"]) / (
        "ManiSkill2_real2sim/mani_skill2_real2sim/assets/descriptions/widowx_description"
    )
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "vpvla_planner"
    dst = cache / "widowx_description"
    if not dst.exists():
        cache.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
    return dst


class PlanResult:
    def __init__(self, success: bool, path: Optional[np.ndarray], plan_time: float, status: str):
        self.success = success
        self.path = path  # (N, 6) joint waypoints, first row is the start
        self.plan_time = plan_time
        self.status = status


class MPlibPlanner:
    name = "mplib"

    def __init__(self, planning_time: float = 2.0, cloud_voxel: float = 0.01):
        import mplib

        self._mplib = mplib
        urdf = widowx_asset_dir() / "wx250s.urdf"
        self.planner = mplib.Planner(urdf=str(urdf), move_group=EE_LINK)
        self._ee_idx = self.planner.user_link_names.index(EE_LINK)
        self.planning_time = planning_time
        self.cloud_voxel = cloud_voxel

    def fk(self, q_arm: np.ndarray) -> np.ndarray:
        model = self.planner.pinocchio_model
        model.compute_forward_kinematics(np.concatenate([q_arm[:ARM_DOF], [FINGER_OPEN, FINGER_OPEN]]))
        pose = model.get_link_pose(self._ee_idx)
        return np.concatenate([pose.p, pose.q])

    def plan(self, qpos: np.ndarray, goal_pose: np.ndarray, view: CameraView, robot_ids) -> PlanResult:
        cloud = view.obstacle_cloud(robot_ids, voxel=self.cloud_voxel, workspace=WORKSPACE)
        start = time.perf_counter()
        self.planner.update_point_cloud(cloud, resolution=self.cloud_voxel)
        q = np.concatenate([qpos[:ARM_DOF], [FINGER_OPEN, FINGER_OPEN]])
        goal = self._mplib.Pose(goal_pose[:3], goal_pose[3:])
        result = self.planner.plan_pose(goal, q, time_step=0.1, planning_time=self.planning_time)
        elapsed = time.perf_counter() - start
        if result["status"] != "Success":
            return PlanResult(False, None, elapsed, str(result["status"]))
        return PlanResult(True, np.asarray(result["position"])[:, :ARM_DOF], elapsed, "Success")


class CuroboPlanner:
    name = "curobo"

    def __init__(self, voxel_size: float = 0.01, device: str = "cuda:0"):
        import torch
        import yaml

        from curobo.perception import Mapper, MapperCfg

        self._torch = torch
        self.device = device
        config_path = Path(__file__).with_name("widowx_curobo.yml")
        robot = yaml.safe_load(config_path.read_text())
        assets = widowx_asset_dir()
        robot["kinematics"]["urdf_path"] = str(assets / "wx250s.urdf")
        robot["kinematics"]["asset_root_path"] = str(assets)
        self._robot_cfg = robot

        lo, hi = np.asarray(WORKSPACE[0]), np.asarray(WORKSPACE[1])
        self.mapper = Mapper(MapperCfg(
            extent_meters_xyz=tuple((hi - lo).tolist()),
            extent_esdf_meters_xyz=tuple((hi - lo).tolist()),
            grid_center=torch.tensor((hi + lo) / 2, dtype=torch.float32),
            voxel_size=voxel_size,
            esdf_voxel_size=voxel_size,
            truncation_distance=voxel_size * 4,
            depth_minimum_distance=0.05,
            depth_maximum_distance=2.0,
            num_cameras=1,
            image_height=480,
            image_width=640,
            block_size=2,
            device=device,
        ))
        self.planner = None  # built on the first plan, once there's a map to size the collision cache

    def _build_planner(self, scene):
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

        cfg = MotionPlannerCfg.create(robot=self._robot_cfg, scene_model=scene)
        self.planner = MotionPlanner(cfg)
        self.planner.warmup(enable_graph=True, num_warmup_iterations=2)

    def _map(self, view: CameraView, robot_ids):
        from curobo.scene import Scene
        from curobo.types import CameraObservation, Pose

        torch = self._torch
        depth = view.depth.copy()
        depth[~view.obstacle_mask(robot_ids)] = 0.0  # cut the robot (and anything it holds) out
        cam_pose = mat_to_pose(view.base_from_cam)
        obs = CameraObservation(
            depth_image=torch.tensor(depth, dtype=torch.float32, device=self.device).unsqueeze(0),
            rgb_image=torch.tensor(view.rgb, dtype=torch.uint8, device=self.device).unsqueeze(0),
            intrinsics=torch.tensor(view.intrinsic, dtype=torch.float32, device=self.device).unsqueeze(0),
            pose=Pose(
                position=torch.tensor(cam_pose[None, :3], dtype=torch.float32, device=self.device),
                quaternion=torch.tensor(cam_pose[None, 3:], dtype=torch.float32, device=self.device),
            ),
        )
        self.mapper.reset()
        self.mapper.integrate(camera_observation=obs)
        return Scene(voxel=[self.mapper.compute_esdf()])

    def fk(self, q_arm: np.ndarray) -> np.ndarray:
        from curobo.types import JointState

        torch = self._torch
        if self.planner is None:
            raise RuntimeError("fk() needs a planner; call plan() first")
        q = torch.tensor(q_arm[None, :ARM_DOF], dtype=torch.float32, device=self.device)
        state = self.planner.compute_kinematics(JointState.from_position(q, joint_names=self.planner.joint_names))
        pose = state.tool_poses.get_link_pose(EE_LINK)
        return np.concatenate([pose.position[0].cpu().numpy(), pose.quaternion[0].cpu().numpy()]).astype(np.float64)

    def plan(self, qpos: np.ndarray, goal_pose: np.ndarray, view: CameraView, robot_ids) -> PlanResult:
        from curobo.types import GoalToolPose, JointState

        torch = self._torch
        start = time.perf_counter()
        scene = self._map(view, robot_ids)
        if self.planner is None:
            self._build_planner(scene)
            start = time.perf_counter()  # don't count one-time warmup as planning time
            scene = self._map(view, robot_ids)
        # Overwrite the map's values in place: the planner's CUDA graphs keep pointing at this
        # memory, and update_world() would swap in a new grid they can't see.
        grid = scene.voxel[0]
        self.planner.scene_collision_checker.data.voxels.update_features(grid.feature_tensor.reshape(-1, 1), grid.name)
        q = torch.tensor(qpos[None, :ARM_DOF], dtype=torch.float32, device=self.device)
        goal = GoalToolPose(
            tool_frames=self.planner.tool_frames,
            position=torch.tensor(goal_pose[:3], dtype=torch.float32, device=self.device).view(1, 1, 1, 1, 3),
            quaternion=torch.tensor(goal_pose[3:], dtype=torch.float32, device=self.device).view(1, 1, 1, 1, 4),
        )
        result = self.planner.plan_pose(goal, JointState.from_position(q, joint_names=self.planner.joint_names))
        elapsed = time.perf_counter() - start
        if result is None or not bool(result.success.any()):
            status = getattr(result, "status", None) if result is not None else "no result"
            return PlanResult(False, None, elapsed, str(status))
        plan = result.get_interpolated_plan()
        cols = [plan.joint_names.index(name) for name in self.planner.joint_names]  # drops the locked finger
        path = plan.position.reshape(-1, plan.position.shape[-1])[:, cols].cpu().numpy()
        return PlanResult(True, path.astype(np.float64), elapsed, "Success")


def make_planner(name: str):
    if name == "mplib":
        return MPlibPlanner()
    if name == "curobo":
        return CuroboPlanner()
    raise ValueError(f"unknown planner '{name}'")
