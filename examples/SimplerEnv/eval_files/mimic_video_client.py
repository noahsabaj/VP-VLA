"""The evaluator's model interface for mimic-video (the experiments README, "mimic-video").

It stands where ModelClientVP stands for VP-VLA (`--policy-client mimic_video`), for every controller that has a
learned policy. mimic-video's policy runs in its own server (mimic_video_server.py), in its own environment; this
client only carries each step's picture and the robot's state to it and the simulator's action back. mimic-video
reads what VP-VLA does not: the hand's pose in the robot's base frame and the gripper's opening, which this client
takes from the simulator exactly as mimic-video's own eval loop does. It uses no visual prompts and no VLM.

The requests' round trips count as the policy's ("vla") in the episode's cost record (model_time.py). One step in
five asks the model for a new action chunk; the others only read the chunk.
"""

import uuid
from typing import Optional

import numpy as np

from examples.SimplerEnv.eval_files.model_time import time_requests

# The command of mimic-video's policy is a change of the controller's target pose, so it holds for this mode only.
CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
# The gripper's opening as mimic-video's eval loop scales it: the fingers' mean joint position, 0 closed to 1 open.
GRIPPER_CLOSED, GRIPPER_OPEN = 0.015, 0.037


def robot_state(env) -> dict:
    """The hand's pose in the robot's base frame (position, quaternion with w first) and the gripper's opening."""
    agent = env.unwrapped.agent
    if agent.control_mode != CONTROL_MODE:
        raise ValueError(f"mimic-video's actions are for the control mode {CONTROL_MODE}, not {agent.control_mode}")
    controllers = agent.controllers[CONTROL_MODE].controllers
    pose = controllers["arm"].ee_pose_at_base
    gripper = (controllers["gripper"].qpos.mean() - GRIPPER_CLOSED) / (GRIPPER_OPEN - GRIPPER_CLOSED)
    return {"ee_p": np.asarray(pose.p), "ee_q": np.asarray(pose.q), "gripper": float(gripper)}


class MimicVideoClient:
    """reset(), step() and visualize_epoch() as the evaluator calls them; step() needs the simulator (env)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 10093, client=None) -> None:
        if client is None:
            from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

            client = time_requests(WebsocketClientPolicy(host, port), "vla")
        self.client = client
        self.session: Optional[str] = None
        self.task_description: Optional[str] = None
        self.chunks = 0

    def _ask(self, **request) -> dict:
        answer = self.client.predict_action({"session": self.session, **request})
        if not answer.get("ok"):
            raise RuntimeError(f"mimic-video's server: {answer.get('error', answer)}")
        return answer["data"]

    def _end(self) -> None:
        if self.session is not None:
            self._ask(op="end")
            self.session = None

    def reset(self, task_description: str) -> None:
        self._end()
        self.session = uuid.uuid4().hex
        self.task_description = task_description
        self.chunks = 0
        self._ask(op="reset", task=task_description)

    def step(self, image: np.ndarray, task_description: Optional[str] = None, *args, env=None, **kwargs):
        """(raw_action, action): the action for the simulator, and the same numbers under the names of the
        evaluator's action plots."""
        if env is None:
            raise ValueError("mimic-video reads the robot's state: step() needs the simulator (env=...)")
        if self.session is None or (task_description is not None and task_description != self.task_description):
            self.reset(self.task_description if task_description is None else task_description)
        answer = self._ask(op="step", image=np.ascontiguousarray(image), task=self.task_description,
                           **robot_state(env))
        self.chunks += bool(answer["new_chunk"])
        action = {k: answer[k] for k in ("world_vector", "rot_axangle", "gripper", "terminate_episode")}
        raw_action = {"world_vector": action["world_vector"], "rotation_delta": action["rot_axangle"],
                      "open_gripper": (action["gripper"] + 1.0) / 2.0}
        return raw_action, action

    def finish_episode(self, success: bool, info: dict) -> None:
        self._end()

    def visualize_epoch(self, predicted_raw_actions, images, save_path: str) -> None:
        """The episode's actions over its steps, as a picture."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        rows = [("world_vector", ("x", "y", "z")), ("rotation_delta", ("rx", "ry", "rz")), ("open_gripper", ("open",))]
        fig, axes = plt.subplots(len(rows), 1, figsize=(8, 6), sharex=True)
        for ax, (key, names) in zip(axes, rows, strict=True):
            values = np.array([np.ravel(a[key]) for a in predicted_raw_actions])
            for i, name in enumerate(names):
                ax.plot(values[:, i], label=name)
            ax.set_ylabel(key)
            ax.legend(loc="upper right", fontsize=7)
        axes[-1].set_xlabel("step")
        fig.tight_layout()
        fig.savefig(save_path, dpi=100)
        plt.close(fig)
