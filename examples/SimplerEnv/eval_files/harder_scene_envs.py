"""The harder scenes of harder_scenes.py and the new test scenes of novel_scenes.py as ManiSkill2_real2sim tasks,
registered under their names when this file is imported (planner_evaluator.py imports it). Needs the simulator.

Each task is SimplerEnv's PutOnBridgeInSceneEnv (the spoon and carrot tasks' table, camera, lighting and robot) with
all of the scene's objects instead of two; a sink scene is SimplerEnv's PutEggplantInBasketScene (the eggplant task's
sink, camera and robot) in the same way. Each goal is checked as SimplerEnv's PutOnInSceneEnv.evaluate checks its one
goal, for any pair of objects. Objects 0 and 1 stay the source and target object that SimplerEnv's code and the
planner's logs read. The new test scenes have no fixed layouts: planner_evaluator.py gives each episode its own.
"""

import json
from collections import OrderedDict

import numpy as np
from mani_skill2_real2sim import ASSET_DIR
from mani_skill2_real2sim.envs.custom_scenes.put_on_in_scene import PutEggplantInBasketScene, PutOnBridgeInSceneEnv
from mani_skill2_real2sim.utils.registration import register_env
from transforms3d.quaternions import quat2mat

from . import harder_scenes, novel_scenes
from .harder_scenes import Task

TASKS = {**harder_scenes.TASKS, **novel_scenes.TASKS}
# The model infos of SimplerEnv's Google-robot objects, which a Bridge scene does not load (novel_scenes.PICK_MODELS).
PICK_MODEL_JSON = ASSET_DIR / "custom" / "info_pick_custom_v0.json"


def layouts(name: str):
    """The task's xy and turn configs: a harder scene's fixed layouts; for a new test scene, one random layout that
    only stands in until the evaluator gives the episode its own (novel_scenes.random_layout)."""
    if name in harder_scenes.TASKS:
        return harder_scenes.layouts(name)
    xy, quats = novel_scenes.random_layout(name, 0, 0)
    return [xy], [quats]


class HarderBridgeScene(PutOnBridgeInSceneEnv):
    TASK_NAME: str
    OVERLAY_OBJECTS = ()  # objects drawn from the real picture, as the eggplant task's sink (rgb_always_overlay_...)

    def __init__(self, **kwargs):
        self.task: Task = TASKS[self.TASK_NAME]  # before __init__, which already resets the scene once
        xys, quats = layouts(self.TASK_NAME)
        picks = [m for m in self.task.objects if m in novel_scenes.PICK_MODELS]
        if picks:
            db = json.loads(PICK_MODEL_JSON.read_text())
            kwargs["model_db_override"] = {**{m: db[m] for m in picks}, **kwargs.get("model_db_override", {})}
        if self.OVERLAY_OBJECTS:
            kwargs.setdefault("rgb_always_overlay_objects", list(self.OVERLAY_OBJECTS))
        # PutOnBridgeInSceneEnv's own __init__, also in a sink scene (PutEggplantInBasketScene's sets the eggplant).
        PutOnBridgeInSceneEnv.__init__(self, source_obj_name=self.task.objects[0], target_obj_name=self.task.objects[1],
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
        on = {p: self._on(*p, z_offset=self.task.z_offset) for p in self.task.goals + self.task.logged}
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


class SinkScene(HarderBridgeScene, PutEggplantInBasketScene):
    """The eggplant task's sink scene: PutEggplantInBasketScene adds the sink (_load_model), moves it into place
    (_initialize_actors) and gives the robot and camera of its prepackaged config; the rest is HarderBridgeScene's."""
    OVERLAY_OBJECTS = ("sink", "dummy_sink_target_plane")


def _register(name: str) -> None:
    base = SinkScene if TASKS[name].scene == "sink" else HarderBridgeScene
    cls = type(name.split("-")[0], (base,), {"TASK_NAME": name, "__module__": __name__})
    globals()[cls.__name__] = cls  # so the gym entry point "harder_scene_envs:<class>" resolves
    register_env(name, max_episode_steps=TASKS[name].max_steps)(cls)


for _name in TASKS:
    _register(_name)
