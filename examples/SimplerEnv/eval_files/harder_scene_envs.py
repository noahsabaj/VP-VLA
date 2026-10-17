"""The harder scenes of harder_scenes.py as ManiSkill2_real2sim tasks, registered under their names when this file is
imported (planner_evaluator.py imports it). Needs the simulator.

Each task is SimplerEnv's PutOnBridgeInSceneEnv (the spoon and carrot tasks' table, camera, lighting and robot) with
all of the scene's objects instead of two. Each goal is checked as SimplerEnv's PutOnInSceneEnv.evaluate checks its
one goal, for any pair of objects. Objects 0 and 1 stay the source and target object that SimplerEnv's code and the
planner's logs read.
"""

from collections import OrderedDict

import numpy as np
from mani_skill2_real2sim.envs.custom_scenes.put_on_in_scene import PutOnBridgeInSceneEnv
from mani_skill2_real2sim.utils.registration import register_env
from transforms3d.quaternions import quat2mat

from .harder_scenes import TASKS, Task, layouts


class HarderBridgeScene(PutOnBridgeInSceneEnv):
    TASK_NAME: str

    def __init__(self, **kwargs):
        self.task: Task = TASKS[self.TASK_NAME]  # before super().__init__, which already resets the scene once
        xys, quats = layouts(self.TASK_NAME)
        super().__init__(source_obj_name=self.task.objects[0], target_obj_name=self.task.objects[1],
                         xy_configs=xys, quat_configs=quats, **kwargs)

    def reset(self, seed=None, options=None):
        """PutOnBridgeInSceneEnv.reset with all of the scene's objects."""
        options = dict(options or {})
        self.set_episode_rng(seed)
        obj_init_options = dict(options.get("obj_init_options", {}))
        n, q = len(self._xy_configs) * len(self._quat_configs), len(self._quat_configs)
        episode_id = obj_init_options.get("episode_id", self._episode_rng.randint(n))
        obj_init_options.update(source_obj_id=0, target_obj_id=1, init_xys=self._xy_configs[(episode_id % n) // q],
                                init_rot_quats=self._quat_configs[episode_id % q])
        options["model_ids"] = list(self.task.objects)
        options["obj_init_options"] = obj_init_options
        obs, info = super(PutOnBridgeInSceneEnv, self).reset(seed=self._episode_seed, options=options)
        info.update({"episode_id": episode_id})
        return obs, info

    def _load_model(self):
        super()._load_model()
        # Two actors of one model (the two plates) get different names: SimplerEnv's code tells objects apart by name.
        seen = {}
        for obj in self.episode_objs:
            k = seen[obj.name] = seen.get(obj.name, 0) + 1
            if k > 1:
                obj.name = f"{obj.name}_{k}"

    def _initialize_actors(self):
        super()._initialize_actors()
        # Every object's box in the world frame after settling, as SimplerEnv's code finds it for its two objects.
        sizes = zip(self.episode_objs, self.episode_model_bbox_sizes, strict=True)
        self.episode_obj_bboxes_world = [quat2mat(obj.pose.q) @ size for obj, size in sizes]

    def _pair_name(self, pair) -> str:
        return f"{self.task.names[pair[0]]}_on_{self.task.names[pair[1]]}"

    def _initialize_episode_stats(self):
        self.episode_stats = OrderedDict((self._pair_name(p), False) for p in self.task.goals + self.task.logged)
        self.episode_stats["moved_other_obj"] = False

    def _on(self, source: int, target: int, completely: bool, z_offset: float = 0.02) -> bool:
        """PutOnInSceneEnv.evaluate's check that the source lies on the target: its centre within the target's box
        (seen from above) and just above it; with `completely`, also touching nothing but the target and the robot."""
        src, tgt = self.episode_objs[source], self.episode_objs[target]
        offset = src.pose.transform(src.cmass_local_pose).p - tgt.pose.transform(tgt.cmass_local_pose).p
        half_src, half_tgt = self.episode_obj_bboxes_world[source] / 2, self.episode_obj_bboxes_world[target] / 2
        if np.linalg.norm(offset[:2]) > np.linalg.norm(half_tgt[:2]) + 0.003:
            return False
        if not (offset[2] > 0 and offset[2] - half_tgt[2] - half_src[2] <= z_offset):
            return False
        if not completely:
            return True
        allowed = {tgt.id} | {link.id for link in self.agent.robot.get_links()}
        for contact in self._scene.get_contacts():
            ids = (contact.actor0.id, contact.actor1.id)
            if src.id not in ids:
                continue
            other = ids[1] if ids[0] == src.id else ids[0]
            if other not in allowed and np.linalg.norm(np.sum([p.impulse for p in contact.points], axis=0)) > 1e-6:
                return False
        return True

    def evaluate(self, **kwargs):
        on = {p: self._on(*p) for p in self.task.goals + self.task.logged}
        moving = {i for p in self.task.goals for i in p[:2]}
        starts = enumerate(zip(self.episode_objs, self.episode_obj_xyzs_after_settle, strict=True))
        moved_other = any(np.linalg.norm(np.asarray(xyz)[:2] - obj.pose.p[:2]) > 0.03
                          for i, (obj, xyz) in starts if i not in moving)
        stats = {self._pair_name(p): bool(v) for p, v in on.items()}
        stats["moved_other_obj"] = bool(moved_other)
        self.episode_stats.update(stats)
        return dict(stats, episode_stats=self.episode_stats, success=all(on[p] for p in self.task.goals))

    def get_language_instruction(self, **kwargs):
        return self.task.instruction


def _register(name: str) -> None:
    cls = type(name.split("-")[0], (HarderBridgeScene,), {"TASK_NAME": name, "__module__": __name__})
    globals()[cls.__name__] = cls  # so the gym entry point "harder_scene_envs:<class>" resolves
    register_env(name, max_episode_steps=TASKS[name].max_steps)(cls)


for _name in TASKS:
    _register(_name)
