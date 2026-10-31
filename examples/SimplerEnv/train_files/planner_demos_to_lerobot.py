"""A VP-VLA training set from recorded demonstrations (examples/SimplerEnv/eval_files/planner/demos.py: an eval with
--record-demos, or RECORD_DEMOS=1 in the experiments' runners), in the layout that starVLA's LeRobot v2.0 reader wants
(starVLA/dataloader/gr00t_lerobot/datasets.py), with the keys of its Bridge data config (oxe_bridge):

    python -m examples.SimplerEnv.train_files.planner_demos_to_lerobot --out DATA_ROOT/NAME \
        --stats CHECKPOINT_RUN/dataset_statistics.json [--visual-prompts VP_DIR] [--reword] DEMOS_DIR...

DEMOS_DIR: a folder with recorded episodes at any depth (each one a folder with meta.json), e.g. a run folder. The
script takes every successful episode (--all: every episode), in the order of their folders, and writes into NAME:
- data/chunk-000/episode_000000.parquet, ...: for each step the state (observation.state, 8 numbers), the action (7),
  timestamp, frame_index, episode_index, index and task_index;
- videos/chunk-000/observation.images.image_0/episode_000000.mp4, ...: the recorded pictures;
- meta/info.json, modality.json, episodes.jsonl, tasks.jsonl and stats_gr00t.json, and meta/sources.jsonl (the
  recorded episode of each episode).

stats_gr00t.json scales the actions and states in training. With --stats FILE it is the checkpoint's own Bridge
statistics (oxe_bridge in its dataset_statistics.json), so a fine-tuned action head keeps the pretrained head's scale
(the eval undoes the scale with the training run's own dataset_statistics.json). Training clips the actions to
Bridge's 1st to 99th percentiles, so the script prints how many actions are outside them, in each dimension. --stats
data computes the statistics from the demonstrations instead. Needs numpy and pyarrow (the starVLA environment).

With --visual-prompts VP_DIR it also writes each episode's visual prompts, which VP-VLA's training draws on the camera
picture (starVLA/dataloader/visual_prompt_datasets.py: a crosshair on the object to move, a box around its place):
VP_DIR/NAME/chunk-000/episode_000000.npz, ..., in the format of the data preparation
(data_preparation/SimplerEnv/process_dataset_parallel_oxe.py), so the training config's visual_prompt_dir is VP_DIR.
The masks are the simulator's own (the recorded object maps), not SAM's, and the subtasks and names that the data
preparation asks a VLM for come from the recording (vlm_entries). One pick-and-place per episode. The script prints,
for each task, the share of steps with the crosshair and with the box (a step has none when its object is hidden).

With --reword each recorded episode is written once with its own instruction and once with one training rewording of
each family (rewordings.py; the 4 WidowX tasks only), the same pictures and actions with another instruction. The
visual prompts of every copy mark the objects of the episode's own task.
"""

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from examples.SimplerEnv.train_files.rewordings import FAMILIES, rewording

FPS = 5  # demos.FPS
CHUNK = 1000  # episodes per chunk folder
VIDEO_KEY = "observation.images.image_0"
STATE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]
ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
STAT_NAMES = ("mean", "std", "min", "max", "q01", "q99")
# What SimplerEnv's instructions call the simulator's objects (model names; a scene's second copy ends in _2). The
# eggplant's basket is part of the background picture: its place is SimplerEnv's invisible target plane in it.
WORDS = {"bridge_carrot_generated_modified": "carrot", "bridge_plate_objaverse_larger": "plate",
         "bridge_spoon_generated_modified": "spoon", "table_cloth_generated_shorter": "towel",
         "green_cube_3cm": "green block", "yellow_cube_3cm": "yellow block", "baked_green_cube_3cm": "green block",
         "baked_yellow_cube_3cm": "yellow block", "eggplant": "eggplant", "dummy_sink_target_plane": "yellow basket"}
SUBTASK_SPLIT = r",\s*and\s*|,\s*|\s+and\s+"  # the data preparation's parse_explicit_task
# A subtask with one of these puts something down (" to ": "transfer the carrot to the platter").
PLACE_WORDS = (" on ", " onto ", " upon ", " in ", " into ", " to ")
LABEL_NOTE = "from the recorded episode, not a VLM"


def load_meta(folder: str) -> dict:
    with open(os.path.join(folder, "meta.json")) as fh:
        return json.load(fh)


def find_episodes(roots: List[str], include_failures: bool = False) -> List[str]:
    """The recorded episodes under `roots` (folders with meta.json and episode.npz), sorted by folder: the
    successful ones, or all with `include_failures`. Episodes without steps are left out."""
    found = set()
    for root in roots:
        for dirpath, _, files in os.walk(root):
            if "meta.json" in files and "episode.npz" in files:
                found.add(os.path.normpath(dirpath))
    keep = []
    for folder in sorted(found):
        meta = load_meta(folder)
        if meta["steps"] > 0 and (include_failures or meta["success"]):
            keep.append(folder)
    return keep


def episode_instructions(meta: dict, family: Optional[str] = None) -> List[str]:
    """The instruction of each step of a recorded episode: its own, or its task's training rewording of `family` for
    this episode at every step."""
    if family is None:
        return list(meta["instructions"])
    try:
        return [rewording(meta["env_name"], family, meta["episode_key"])] * meta["steps"]
    except KeyError:
        raise SystemExit(f"{meta['episode_key']}: no training rewordings for {meta['env_name']}") from None


def modality() -> dict:
    """meta/modality.json: the keys of the Bridge data config (data_config.OxeBridgeDataConfig)."""
    return {"state": {n: {"start": i, "end": i + 1} for i, n in enumerate(STATE_NAMES)},
            "action": {n: {"start": i, "end": i + 1} for i, n in enumerate(ACTION_NAMES)},
            "video": {"image_0": {"original_key": VIDEO_KEY}},
            "annotation": {"human.action.task_description": {"original_key": "task_index"}}}


def info(episodes: int, frames: int, tasks: int, height: int, width: int) -> dict:
    """meta/info.json (LeRobot v2.0)."""
    scalar = {"shape": [1], "names": None}
    return {"codebase_version": "v2.0", "robot_type": "widowx", "total_episodes": episodes, "total_frames": frames,
            "total_tasks": tasks, "total_videos": episodes, "total_chunks": (episodes + CHUNK - 1) // CHUNK,
            "chunks_size": CHUNK, "fps": FPS, "splits": {"train": f"0:{episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                VIDEO_KEY: {"dtype": "video", "shape": [height, width, 3], "names": ["height", "width", "channel"],
                            "video_info": {"video.fps": float(FPS), "video.codec": "h264", "video.pix_fmt": "yuv420p",
                                           "video.is_depth_map": False, "has_audio": False}},
                "observation.state": {"dtype": "float64", "shape": [len(STATE_NAMES)], "names": STATE_NAMES},
                "action": {"dtype": "float64", "shape": [len(ACTION_NAMES)], "names": ACTION_NAMES},
                "timestamp": {"dtype": "float32", **scalar},
                **{k: {"dtype": "int64", **scalar} for k in ("frame_index", "episode_index", "index", "task_index")}}}


def checkpoint_stats(path: str, tag: str = "oxe_bridge") -> dict:
    """meta/stats_gr00t.json from a training run's dataset_statistics.json: its statistics of `tag`."""
    with open(path) as fh:
        s = json.load(fh)[tag]
    return {"action": {k: list(s["action"][k]) for k in STAT_NAMES},
            "observation.state": {k: list(s["state"][k]) for k in STAT_NAMES}}


def data_stats(actions: np.ndarray, states: np.ndarray) -> dict:
    """meta/stats_gr00t.json from the demonstrations themselves."""
    def one(x):
        return {"mean": x.mean(0), "std": x.std(0), "min": x.min(0), "max": x.max(0),
                "q01": np.quantile(x, 0.01, axis=0), "q99": np.quantile(x, 0.99, axis=0)}
    return {"action": {k: v.tolist() for k, v in one(actions).items()},
            "observation.state": {k: v.tolist() for k, v in one(states).items()}}


def clip_report(actions: np.ndarray, stats: dict) -> List[str]:
    """For each moving dimension of the actions: the share below the 1st and above the 99th percentile of `stats`
    (training clips them there) and the largest value."""
    q01, q99 = np.array(stats["action"]["q01"]), np.array(stats["action"]["q99"])
    lines = []
    for i, name in enumerate(ACTION_NAMES[:6]):
        a = actions[:, i]
        lines.append(f"action.{name}: {100 * np.mean(a < q01[i]):.1f}% below {q01[i]:+.4f}, "
                     f"{100 * np.mean(a > q99[i]):.1f}% above {q99[i]:+.4f} (largest |value| {np.abs(a).max():.4f})")
    return lines


def object_word(model: str) -> str:
    return WORDS.get(re.sub(r"_\d+$", "", model), model)


def is_placing(subtask: str) -> bool:
    return any(word in f" {subtask} " for word in PLACE_WORDS)


def vlm_entries(instruction: str, actions: np.ndarray, object_name: str, place_name: str) -> List[dict]:
    """The data preparation's VLM answers (vlm_results), from the recording instead of a VLM: an entry at the first
    step and at each change of the gripper (open: the action's last number above 0.5). The subtasks are the
    instruction's parts (the data preparation's split at commas and "and"). A part that names a place ("... on the
    plate") puts something down and also names the place, as the VLM's prompt asks. Each grasp ends a pick part and
    each release a place part (a failed grasp too), and the next entry has the next part. Training reads only each
    entry's frame and subtask and whether it names an object and a place."""
    parts = [p.strip() for p in re.split(SUBTASK_SPLIT, instruction.lower()) if p.strip()] or [instruction]
    entries, gripper_before, part = [], None, 0
    names = {"target_object": None, "target_location": None}
    for t, action in enumerate(actions):
        gripper = "OPEN" if action[6] > 0.5 else "CLOSED"
        if gripper == gripper_before:
            continue
        changed = gripper_before is not None
        decision = "continue"
        if changed and is_placing(parts[part]) == (gripper == "OPEN") and part + 1 < len(parts):
            part, decision = part + 1, "proceed"
        names["target_object"] = object_name
        if is_placing(parts[part]):
            names["target_location"] = place_name
        entries.append({"frame": t, "subtask": parts[part], "subtask_idx": part, "decision": decision, **names,
                        "reasoning": LABEL_NOTE, "gripper_state_changed": changed})
        gripper_before = gripper
    return entries


def mask_boxes(masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Each mask's box around its pixels (x1, y1, x2, y2) and its score: 1 with pixels, 0 without."""
    boxes = np.zeros((len(masks), 4), np.float32)
    for t, mask in enumerate(masks):
        rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
        if len(rows):
            boxes[t] = [cols[0], rows[0], cols[-1] + 1, rows[-1] + 1]
    return boxes, masks.any((1, 2)).astype(np.float32)


def prompt_masks(object_map: np.ndarray, entries: List[dict], goal: List[int]) -> Dict[str, np.ndarray]:
    """The masks, boxes and scores of the object to move (target_*) and of its place (location_*) at each step: the
    object map's pixels of the goal's object and place, from the entry on that names them."""
    steps = len(object_map)
    named = {key: np.zeros(steps, bool) for key in ("target_object", "target_location")}
    ends = [e["frame"] for e in entries[1:]] + [steps]
    for entry, end in zip(entries, ends, strict=True):
        for key, on in named.items():
            on[entry["frame"]:end] = entry[key] is not None
    out = {}
    for prefix, key, index in (("target", "target_object", goal[0]), ("location", "target_location", goal[1])):
        masks = ((object_map == index + 1) & named[key][:, None, None]).astype(np.uint8)
        out[f"{prefix}_masks"] = masks
        out[f"{prefix}_boxes"], out[f"{prefix}_scores"] = mask_boxes(masks)
    return out


def episode_visual_prompts(folder: str, meta: dict, actions: np.ndarray, object_map: Optional[np.ndarray],
                           instruction: str, episode: int) -> Dict[str, np.ndarray]:
    """One episode's visual-prompt file (the data preparation's npz)."""
    if object_map is None:
        raise SystemExit(f"{folder}: recorded without object maps")
    if len(meta["goals"]) != 1:
        raise SystemExit(f"{folder}: {len(meta['goals'])} goals; the visual prompts take one pick-and-place")
    goal = meta["goals"][0]
    entries = vlm_entries(instruction, actions, object_word(meta["objects"][goal[0]]),
                          object_word(meta["objects"][goal[1]]))
    return {"episode_index": np.array(episode), "task_description": np.array(instruction),
            "num_frames": np.array(len(actions)), "fps": np.array(float(FPS)),
            "target_object": np.array(entries[-1]["target_object"] or ""),
            "target_location": np.array(entries[-1]["target_location"] or ""),
            **prompt_masks(object_map, entries, goal), "vlm_results": np.array(json.dumps(entries))}


def write_parquet(path: str, states: np.ndarray, actions: np.ndarray, episode: int, first_index: int,
                  task_index: List[int]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    steps = len(actions)
    floats = pa.list_(pa.float64())
    table = pa.table({
        "observation.state": pa.array(states.astype(np.float64).tolist(), type=floats),
        "action": pa.array(actions.astype(np.float64).tolist(), type=floats),
        "timestamp": pa.array(np.arange(steps, dtype=np.float32) / FPS, type=pa.float32()),
        "frame_index": pa.array(np.arange(steps), type=pa.int64()),
        "episode_index": pa.array(np.full(steps, episode), type=pa.int64()),
        "index": pa.array(np.arange(first_index, first_index + steps), type=pa.int64()),
        "task_index": pa.array(task_index, type=pa.int64()),
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(table, path)


def write_dataset(out: str, folders: List[str], stats: str, visual_prompts: Optional[str] = None,
                  reword: bool = False) -> Dict[str, object]:
    """The training set of the recorded episodes in `folders` (see the module docstring); `stats` is a
    dataset_statistics.json or "data"; with `visual_prompts` (VP_DIR) also the visual prompts; with `reword` each
    episode also once with a rewording of each family. Returns a summary."""
    vp_out = None if visual_prompts is None else os.path.join(visual_prompts, os.path.basename(os.path.normpath(out)))
    for new in (out, vp_out):
        if new is not None and os.path.exists(new) and os.listdir(new):
            raise SystemExit(f"{new} exists and is not empty")
    if not folders:
        raise SystemExit("no recorded episodes")
    items = [(folder, family) for folder in folders for family in ((None, *FAMILIES) if reword else (None,))]
    tasks: Dict[str, int] = {}
    all_actions, all_states, episodes, sources = [], [], [], []
    shown = defaultdict(lambda: np.zeros(3))  # for each task: steps, steps with the crosshair, steps with the box
    size = None
    frames = 0
    for episode, (folder, family) in enumerate(items):
        meta = load_meta(folder)
        with np.load(os.path.join(folder, "episode.npz")) as z:
            actions, states = z["actions"], z["states"]
            object_map = z["object_map"] if vp_out is not None and "object_map" in z.files else None
        steps = len(actions)
        instructions = episode_instructions(meta, family)
        if len(instructions) != steps or len(states) != steps or meta["steps"] != steps:
            raise SystemExit(f"{folder}: {steps} actions, {len(states)} states, {len(instructions)} instructions")
        if size is None:
            size = (meta["height"], meta["width"])
        elif (meta["height"], meta["width"]) != size:
            raise SystemExit(f"{folder}: pictures of {meta['height']} x {meta['width']}, not {size[0]} x {size[1]}")
        task_index = [tasks.setdefault(t, len(tasks)) for t in instructions]
        chunk = episode // CHUNK
        write_parquet(os.path.join(out, f"data/chunk-{chunk:03d}/episode_{episode:06d}.parquet"), states, actions,
                      episode, frames, task_index)
        video = os.path.join(out, f"videos/chunk-{chunk:03d}/{VIDEO_KEY}/episode_{episode:06d}.mp4")
        os.makedirs(os.path.dirname(video), exist_ok=True)
        shutil.copyfile(os.path.join(folder, "frames.mp4"), video)
        if vp_out is not None:
            if object_map is not None and object_map.shape != (steps, *size):
                raise SystemExit(f"{folder}: object maps of {object_map.shape}, not {(steps, *size)}")
            vp = episode_visual_prompts(folder, meta, actions, object_map, instructions[0], episode)
            path = os.path.join(vp_out, f"chunk-{chunk:03d}", f"episode_{episode:06d}.npz")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(path, **vp)
            shown[meta.get("env_name", "?")] += [steps, vp["target_scores"].sum(), vp["location_scores"].sum()]
        episodes.append({"episode_index": episode, "tasks": list(dict.fromkeys(instructions)), "length": steps})
        sources.append({"episode_index": episode, "source": folder, "episode_key": meta.get("episode_key"),
                        "success": meta["success"], "rewording": family})
        all_actions.append(actions)
        all_states.append(states)
        frames += steps
    actions, states = np.concatenate(all_actions), np.concatenate(all_states)
    stat = data_stats(actions, states) if stats == "data" else checkpoint_stats(stats)
    meta_dir = os.path.join(out, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    files = {"info.json": info(len(items), frames, len(tasks), *size), "modality.json": modality(),
             "stats_gr00t.json": stat}
    for name, value in files.items():
        with open(os.path.join(meta_dir, name), "w") as fh:
            json.dump(value, fh, indent=1)
    lines = {"episodes.jsonl": episodes, "sources.jsonl": sources,
             "tasks.jsonl": [{"task_index": i, "task": t} for t, i in tasks.items()]}
    for name, rows in lines.items():
        with open(os.path.join(meta_dir, name), "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
    prompts = [f"visual prompts, {task}: crosshair {100 * a / n:.1f}% of {n:.0f} steps, box {100 * b / n:.1f}%"
               for task, (n, a, b) in sorted(shown.items())]
    return {"recorded": len(folders), "episodes": len(items), "frames": frames, "tasks": len(tasks),
            "clip": clip_report(actions, stat), "visual_prompts": prompts}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demos", nargs="+", help="folders with recorded episodes")
    ap.add_argument("--out", required=True, help="the new training set's folder (DATA_ROOT/NAME)")
    ap.add_argument("--stats", required=True, help='a training run\'s dataset_statistics.json, or "data"')
    ap.add_argument("--all", action="store_true", help="also the failed episodes")
    ap.add_argument("--visual-prompts", metavar="VP_DIR", help="also write the visual prompts into VP_DIR/NAME")
    ap.add_argument("--reword", action="store_true", help="also each episode with a rewording of each family")
    a = ap.parse_args()
    summary = write_dataset(a.out, find_episodes(a.demos, a.all), a.stats, a.visual_prompts, a.reword)
    print(f"{a.out}: {summary['episodes']} episodes of {summary['recorded']} recorded, {summary['frames']} steps, "
          f"{summary['tasks']} instructions")
    print("\n".join(summary["clip"] + summary["visual_prompts"]))


if __name__ == "__main__":
    main()
