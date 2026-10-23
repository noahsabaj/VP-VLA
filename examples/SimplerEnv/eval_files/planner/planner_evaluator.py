"""Our copy of SimplerEnv's test loop (simpler_env/evaluation/maniskill2_evaluator.py, commit 06accac).

The changes, marked "CHANGED":
1. `model.step()` also gets the full observation and the simulator (`obs=obs, env=env`),
   so a planner can see depth, camera settings and joint angles. ModelClientVP ignores them.
2. After each episode, `model.finish_episode(success, info)` is called if the model has it.
3. `layout_seed` (optional): swap the task's 24 fixed layouts for new random ones, so results
   can be checked on layouts nothing was tuned on. Off by default. With `layout_per_episode`, each
   episode gets a layout of its own instead of one of 12 placements x 4 turns. For the two-object tasks, and for
   the harder scenes with a layout per episode (harder_scenes.random_layout).
4. Before each episode, `model.set_episode(key)` is called if the model has it, so the planner can
   seed its depth noise per episode.
5. `action_plots=False` (`--no-action-plots`) skips the per-episode action plot. On by default.
6. `rgb_noise` (`--rgb-noise`): a real color camera's errors (camera_noise.py) in the picture, seeded per episode
   like the depth noise. The noisy picture is what the model gets and also replaces the camera's picture in `obs`,
   so the VLA and the planner see the same pixels. Off by default.
7. `instruction_set` (`--instruction-set`): a reworded instruction for the task (instruction_sets.py). A given
   instruction now stays for the whole episode; SimplerEnv's loop replaced it with the task's own after the first
   step. Off by default.
8. The harder scenes (harder_scene_envs.py) are registered on import, so `env_name` can be one of them.
Everything else is unchanged, so the AI-only setup behaves exactly as with SimplerEnv's loop.
"""

import os

import numpy as np
from transforms3d.euler import quat2euler

from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video

from .. import harder_scene_envs  # noqa: F401 (registers the harder scenes)
from .. import harder_scenes
from ..camera_noise import RGBNoise
from ..instruction_sets import instruction_for


def run_maniskill2_eval_single_episode(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./results",
    layout_seed=None,
    action_plots=True,
    layout_per_episode=False,
    rgb_noise=None,
):

    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    # Create environment
    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
        additional_env_build_kwargs = ray_tracing_dict
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )

    # CHANGED: optionally replace the task's fixed layouts with held-out random ones.
    if layout_per_episode and layout_seed is None:
        raise ValueError("a layout per episode needs a layout seed")
    if layout_seed is not None and env_name in harder_scenes.TASKS:  # a harder scene: its own random layouts
        if not layout_per_episode:
            raise ValueError(f"{env_name}: random layouts of a harder scene are one per episode (--layout-per-episode)")
        xy, quats = harder_scenes.random_layout(env_name, layout_seed, obj_episode_id)
        env.unwrapped._xy_configs, env.unwrapped._quat_configs = [xy], [quats]
    elif layout_seed is not None:
        if len(env.unwrapped._xy_configs[0]) != 2:
            raise ValueError(f"{env_name}: random layouts are only for the tasks with two objects")
        if layout_per_episode:
            episode_layout(env.unwrapped, layout_seed, obj_episode_id)
        else:
            randomize_layouts(env.unwrapped, layout_seed)
    layout_name = layout_label(layout_seed, layout_per_episode)

    # initialize environment
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }
    obs, _ = env.reset(options=env_reset_options)
    # for long-horizon environments, we check if the current subtask is the final subtask
    is_final_subtask = env.is_final_subtask()

    # Obtain language instruction
    if instruction is not None:
        task_description = instruction
    else:
        # get default language instruction
        task_description = env.get_language_instruction()
    print(task_description)

    # CHANGED: name the episode first, so the camera noise and the model can seed their randomness per episode.
    if obj_variation_mode == "xy":  # no episode id here: the object and robot positions name the try
        episode_key = f"{env_name}/xy/{obj_init_x}/{obj_init_y}/{robot_init_x}/{robot_init_y}"
    else:
        episode_key = f"{env_name}/{layout_name}/{obj_episode_id}"
    if rgb_noise is not None:
        rgb_noise.new_episode(episode_key)

    # Initialize logging
    # CHANGED: the camera's picture through the RGB noise.
    image = camera_image(env, obs, obs_camera_name, rgb_noise)
    images = [image]
    predicted_actions = []
    predicted_terminated, done, truncated = False, False, False

    # Initialize model
    if hasattr(model, "set_episode"):
        model.set_episode(episode_key)
    model.reset(task_description)

    timestep = 0
    success = "failure"

    # Step the environment
    while not (predicted_terminated or truncated):
        # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
        # CHANGED: also pass the full observation and the simulator.
        raw_action, action = model.step(image, task_description, obs=obs, env=env)
        predicted_actions.append(raw_action)
        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        if predicted_terminated:
            if not is_final_subtask:
                # advance the environment to the next subtask
                predicted_terminated = False
                env.advance_to_next_subtask()

        # step the environment
        obs, reward, done, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),
        )

        success = "success" if done else "failure"
        # CHANGED: only without a given instruction does the task's own (next subtask's) instruction replace it.
        new_task_description = env.get_language_instruction() if instruction is None else task_description
        if new_task_description != task_description:
            task_description = new_task_description
            print(task_description)
        is_final_subtask = env.is_final_subtask()

        print(timestep, info)

        image = camera_image(env, obs, obs_camera_name, rgb_noise)
        images.append(image)
        timestep += 1

    episode_stats = info.get("episode_stats", {})

    # CHANGED: let the model log its own per-episode record.
    if hasattr(model, "finish_episode"):
        model.finish_episode(success == "success", {
            "env_name": env_name, "obj_episode_id": obj_episode_id, "layout_seed": layout_seed,
            "layout_per_episode": bool(layout_per_episode), "steps": timestep,
            "episode_stats": {k: bool(v) if isinstance(v, (bool, np.bool_)) else v for k, v in episode_stats.items()},
            "layout": episode_layout_truth(env.unwrapped, obj_episode_id) if obj_variation_mode == "episode" else None,
        })

    # save video
    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"
    ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]
    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        video_name = f"{success}_obj_episode_{obj_episode_id}"
    for k, v in episode_stats.items():
        video_name = video_name + f"_{k}_{v}"
    video_name = video_name + ".mp4"
    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"
    r, p, y = quat2euler(robot_init_quat)
    video_path = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
    video_path = os.path.join(logging_dir, video_path)
    write_video(video_path, images, fps=5)

    # save action trajectory
    # CHANGED: optionally skip the action plot.
    if action_plots:
        action_path = video_path.replace(".mp4", ".png")
        action_root = os.path.dirname(action_path) + "/actions/"
        os.makedirs(action_root, exist_ok=True)
        action_path = action_root + os.path.basename(action_path)
        model.visualize_epoch(predicted_actions, images, save_path=action_path)

    return success == "success"


def camera_image(env, obs, camera_name, rgb_noise=None):
    """The camera's picture for the model, through the RGB noise (which also puts the noisy picture into obs)."""
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=camera_name)
    return image if rgb_noise is None else rgb_noise.apply_to_obs(obs, image)


def draw_layouts(xy_configs, quat_configs, rng: np.random.Generator, n_xy: int, n_quat: int,
                 margin: float = 0.03, min_gap: float = 0.08):
    """n_xy object placements and n_quat turns, drawn near a bridge task's original layouts.

    Object positions are drawn uniformly in each object's original area plus `margin`, keeping the
    two objects `min_gap` apart unless the target never moves (eggplant). A turn is a random rotation
    of the object to move about the vertical on top of its first original orientation; the target
    keeps its orientation.
    """
    from transforms3d.euler import euler2quat
    from transforms3d.quaternions import qmult

    xys = np.array(xy_configs)  # (K, 2 objects, 2)
    lo, hi = xys.min(axis=0) - margin, xys.max(axis=0) + margin
    fixed_target = np.ptp(xys[:, 1], axis=0).max() < 1e-6
    if fixed_target:
        lo[1] = hi[1] = xys[0, 1]
    new_xy = []
    while len(new_xy) < n_xy:
        pair = rng.uniform(lo, hi)
        if fixed_target or np.linalg.norm(pair[0] - pair[1]) >= min_gap:
            new_xy.append(pair)
    base = np.array(quat_configs[0])
    new_quat = []
    for _ in range(n_quat):
        turn = euler2quat(0, 0, rng.uniform(-np.pi, np.pi))
        new_quat.append(np.array([qmult(turn, base[0]), base[1]]))
    return new_xy, new_quat


def randomize_layouts(env, seed: int, n_xy: int = 12, n_quat: int = 4, margin: float = 0.03,
                      min_gap: float = 0.08) -> None:
    """Replace a bridge task's layout lists with random ones drawn near the original layouts.

    Episode ids then index n_xy * n_quat new layouts (48 by default): every placement with every
    turn (draw_layouts). Same seed, same layouts.
    """
    env._xy_configs, env._quat_configs = draw_layouts(env._xy_configs, env._quat_configs,
                                                      np.random.default_rng(seed), n_xy, n_quat, margin, min_gap)


def episode_layout(env, seed: int, episode: int, margin: float = 0.03, min_gap: float = 0.08) -> None:
    """Replace a bridge task's layout lists with one random layout for this episode only.

    One placement and one turn, drawn as in randomize_layouts but from a generator seeded with
    (seed, episode): each episode's layout is independent of the other episodes and of the order
    they run in. Same seed and episode, same layout.
    """
    env._xy_configs, env._quat_configs = draw_layouts(env._xy_configs, env._quat_configs,
                                                      np.random.default_rng([seed, episode]), 1, 1, margin, min_gap)


def layout_label(layout_seed, layout_per_episode: bool) -> str:
    """The layout set's part of the episode key (for the depth noise, the planner seed and the
    hidden-state file names): the layout seed; with a layout per episode, "e" and the seed."""
    return f"e{layout_seed}" if layout_per_episode else f"{layout_seed}"


def episode_layout_truth(env, episode: int):
    """Where the episode's two objects start (x, y in m, simulator frame) and the yaw of the object to
    move (rad, about the vertical: its random turn plus its first original yaw); in a scene with more objects,
    also where each starts. Logged for analysis only; no controller reads it."""
    if not hasattr(env, "_xy_configs"):
        return None
    k, q = len(env._xy_configs), len(env._quat_configs)
    xy = np.asarray(env._xy_configs[(episode % (k * q)) // q])
    quat = np.asarray(env._quat_configs[episode % q])
    out = {"source_xy": xy[0].tolist(), "target_xy": xy[1].tolist(), "source_yaw": float(quat2euler(quat[0])[2])}
    if len(xy) > 2:
        out["objects_xy"] = xy.tolist()
    return out


def maniskill2_evaluator(model, args):
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []
    # CHANGED: one RGB camera noise for the run; each episode reseeds it.
    rgb_noise = RGBNoise(getattr(args, "rgb_noise", "none"), getattr(args, "noise_seed", None))

    # run inference
    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                kwargs = dict(
                    model=model,
                    ckpt_path=args.ckpt_path,
                    robot_name=args.robot,
                    env_name=args.env_name,
                    scene_name=args.scene_name,
                    robot_init_x=robot_init_x,
                    robot_init_y=robot_init_y,
                    robot_init_quat=robot_init_quat,
                    control_mode=control_mode,
                    additional_env_build_kwargs=args.additional_env_build_kwargs,
                    rgb_overlay_path=args.rgb_overlay_path,
                    control_freq=args.control_freq,
                    sim_freq=args.sim_freq,
                    max_episode_steps=args.max_episode_steps,
                    enable_raytracing=args.enable_raytracing,
                    additional_env_save_tags=args.additional_env_save_tags,
                    obs_camera_name=args.obs_camera_name,
                    logging_dir=args.logging_dir,
                    layout_seed=getattr(args, "layout_seed", None),
                    action_plots=not getattr(args, "no_action_plots", False),
                    layout_per_episode=getattr(args, "layout_per_episode", False),
                    rgb_noise=rgb_noise,
                    instruction=instruction_for(getattr(args, "instruction_set", "default"), args.env_name),
                )
                if args.obj_variation_mode == "xy":
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            success_arr.append(
                                run_maniskill2_eval_single_episode(
                                    obj_init_x=obj_init_x,
                                    obj_init_y=obj_init_y,
                                    **kwargs,
                                )
                            )
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(args.obj_episode_range[0], args.obj_episode_range[1]):
                        success_arr.append(run_maniskill2_eval_single_episode(obj_episode_id=obj_episode_id, **kwargs))
                else:
                    raise NotImplementedError()

    return success_arr
