"""A VP-VLA training set from recorded demonstrations (examples/SimplerEnv/eval_files/planner/demos.py: an eval with
--record-demos, or RECORD_DEMOS=1 in the experiments' runners), in the layout that starVLA's LeRobot v2.0 reader wants
(starVLA/dataloader/gr00t_lerobot/datasets.py), with the keys of its Bridge data config (oxe_bridge):

    python examples/SimplerEnv/train_files/planner_demos_to_lerobot.py --out DATA_ROOT/NAME \
        --stats CHECKPOINT_RUN/dataset_statistics.json DEMOS_DIR...

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
"""

import argparse
import json
import os
import shutil
from typing import Dict, List

import numpy as np

FPS = 5  # demos.FPS
CHUNK = 1000  # episodes per chunk folder
VIDEO_KEY = "observation.images.image_0"
STATE_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"]
ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
STAT_NAMES = ("mean", "std", "min", "max", "q01", "q99")


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


def episode_instructions(meta: dict) -> List[str]:
    """The instruction of each step of a recorded episode."""
    return list(meta["instructions"])


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


def write_dataset(out: str, folders: List[str], stats: str) -> Dict[str, object]:
    """The training set of the recorded episodes in `folders` (see the module docstring); `stats` is a
    dataset_statistics.json or "data". Returns a summary."""
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit(f"{out} exists and is not empty")
    if not folders:
        raise SystemExit("no recorded episodes")
    tasks: Dict[str, int] = {}
    all_actions, all_states, episodes, sources = [], [], [], []
    size = None
    frames = 0
    for episode, folder in enumerate(folders):
        meta = load_meta(folder)
        with np.load(os.path.join(folder, "episode.npz")) as z:
            actions, states = z["actions"], z["states"]
        steps = len(actions)
        instructions = episode_instructions(meta)
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
        episodes.append({"episode_index": episode, "tasks": list(dict.fromkeys(instructions)), "length": steps})
        sources.append({"episode_index": episode, "source": folder, "episode_key": meta.get("episode_key"),
                        "success": meta["success"]})
        all_actions.append(actions)
        all_states.append(states)
        frames += steps
    actions, states = np.concatenate(all_actions), np.concatenate(all_states)
    stat = data_stats(actions, states) if stats == "data" else checkpoint_stats(stats)
    meta_dir = os.path.join(out, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    files = {"info.json": info(len(folders), frames, len(tasks), *size), "modality.json": modality(),
             "stats_gr00t.json": stat}
    for name, value in files.items():
        with open(os.path.join(meta_dir, name), "w") as fh:
            json.dump(value, fh, indent=1)
    lines = {"episodes.jsonl": episodes, "sources.jsonl": sources,
             "tasks.jsonl": [{"task_index": i, "task": t} for t, i in tasks.items()]}
    for name, rows in lines.items():
        with open(os.path.join(meta_dir, name), "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)
    return {"episodes": len(folders), "frames": frames, "tasks": len(tasks), "clip": clip_report(actions, stat)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("demos", nargs="+", help="folders with recorded episodes")
    ap.add_argument("--out", required=True, help="the new training set's folder (DATA_ROOT/NAME)")
    ap.add_argument("--stats", required=True, help='a training run\'s dataset_statistics.json, or "data"')
    ap.add_argument("--all", action="store_true", help="also the failed episodes")
    a = ap.parse_args()
    summary = write_dataset(a.out, find_episodes(a.demos, a.all), a.stats)
    print(f"{a.out}: {summary['episodes']} episodes, {summary['frames']} steps, {summary['tasks']} instructions")
    print("\n".join(summary["clip"]))


if __name__ == "__main__":
    main()
