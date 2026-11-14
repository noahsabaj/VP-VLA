"""Recorded demonstrations as training episodes for mimic-video's action decoder (the experiments README,
"mimic-video").

    MIMIC_PYTHON -m examples.SimplerEnv.train_files.planner_demos_to_mimic --out DIR --embeddings FILE.npz \\
        [--checkpoints MIMIC_CHECKPOINTS] [--limit N] DEMOS...

DEMOS: folders with demonstrations (planner/demos.py: EPISODE_KEY/frames.mp4, episode.npz and meta.json), searched
for meta.json. Only successful, whole episodes are taken (the planner's part of a VLA-then-planner episode is not).
Run it from this repository's root with the Python of mimic-video's environment (safetensors, OpenCV, and its T5
text encoder, in MIMIC_CHECKPOINTS/text_encoder, for the instructions that FILE.npz does not have yet). It needs no
GPU: the encoder runs on the CPU (about 20 GB of memory) once for each different instruction.

Each episode becomes DIR/NAME.safetensors with the keys of mimic-video's own Bridge data
(data_preprocessing/action/process_bridge.py and precompute_t5.py), which its training reads unchanged
(configs/dataloading/bridge.yaml):
- workspace_rgb (steps x 480 x 640 x 3, uint8): the picture before each step's command;
- eef_pose_lowdim (steps x 4 x 4): the hand's pose in the robot's base frame at that step. mimic-video's action is
  not the command: it is the hand's pose at the next steps (0.2 s later, 15 of them at 5 Hz) relative to its pose
  now, so the recorded poses are all that it needs;
- gripper_state_lowdim (steps): the gripper's opening, 0 closed to 1 open. Measured (the fingers' joint positions,
  scaled as mimic-video's eval loop scales them) when the demonstration has them (`fingers`, recorded since
  2026-11-14); else the last gripper command, which differs from the measured opening while the gripper moves or
  holds an object;
- gripper_action_lowdim (steps): the gripper command in force at that step (the last command before it; 1 open and
  0 closed), as Bridge's desired state is. mimic-video reads its gripper action 0.2 s later too, and the command in
  force one step later is the step's own command;
- language_instruction and language_embedding (1 x 512 x 1024: mimic-video's T5 encoder, one encoding for each
  different instruction, kept in FILE.npz);
- NAME_timestamps for each: the steps' times in nanoseconds, 5 steps a second.
"""

import argparse
import hashlib
import json
import os
import shlex
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from examples.SimplerEnv.eval_files.planner.demos import FPS, euler_to_matrix

S_TO_NS = 1_000_000_000
HEIGHT, WIDTH = 480, 640
# The gripper's opening as mimic-video's eval loop scales it (mimic_video_client.py).
GRIPPER_CLOSED, GRIPPER_OPEN = 0.015, 0.037


def find_episodes(roots: List[str]) -> List[str]:
    """The demonstration folders under `roots` (the folders with a meta.json), sorted."""
    found = set()
    for root in roots:
        for folder, _, files in os.walk(root):
            if "meta.json" in files:
                found.add(os.path.normpath(folder))
    return sorted(found)


def pose_matrices(states: np.ndarray) -> np.ndarray:
    """The hand's pose at each step as a 4 x 4 matrix, from the demonstration's states (position, roll, pitch and
    yaw in the robot's base frame)."""
    out = np.tile(np.eye(4), (len(states), 1, 1))
    for i, s in enumerate(np.asarray(states, np.float64)):
        out[i, :3, :3] = euler_to_matrix(s[3:6])
        out[i, :3, 3] = s[:3]
    return out


def gripper_state(arrays: Dict[str, np.ndarray]) -> Tuple[np.ndarray, str]:
    """(the gripper's opening at each step, 0 closed to 1 open; "measured" or "command")."""
    if "fingers" in arrays:
        fingers = np.asarray(arrays["fingers"], np.float64).mean(axis=1)
        return ((fingers - GRIPPER_CLOSED) / (GRIPPER_OPEN - GRIPPER_CLOSED)).clip(0.0, 1.0), "measured"
    return np.asarray(arrays["states"], np.float64)[:, 7].clip(0.0, 1.0), "command"


def episode_tensors(frames: np.ndarray, arrays: Dict[str, np.ndarray], instruction: str,
                    embedding: Optional[np.ndarray] = None) -> Tuple[Dict[str, np.ndarray], str]:
    """(the episode's file contents, where its gripper state came from)."""
    steps = len(arrays["actions"])
    frames = np.asarray(frames)
    if frames.shape != (steps, HEIGHT, WIDTH, 3) or frames.dtype != np.uint8:
        raise ValueError(f"the pictures are {frames.shape} {frames.dtype}, not {steps} x {HEIGHT} x {WIDTH} x 3 uint8")
    if len(arrays["states"]) != steps:
        raise ValueError(f"{len(arrays['states'])} states for {steps} actions")
    command = np.asarray(arrays["states"], np.float64)[:, 7]  # the command in force: the last one before the step
    if not np.array_equal(command[1:] > 0.5, np.asarray(arrays["actions"])[:-1, 6] > 0.5):
        raise ValueError("the states' gripper commands are not the actions' commands one step later")
    if not instruction.strip():
        raise ValueError("no instruction")
    times = np.arange(steps, dtype=np.float64) * (S_TO_NS / FPS)
    opening, source = gripper_state(arrays)
    data = {
        "workspace_rgb": frames,
        "eef_pose_lowdim": pose_matrices(arrays["states"]),
        "gripper_state_lowdim": opening,
        "gripper_action_lowdim": (command > 0.5).astype(np.float64),
    }
    data.update({f"{key}_timestamps": times.copy() for key in list(data)})
    data["language_instruction"] = np.frombuffer(instruction.encode("utf-8"), dtype=np.uint8)[None].copy()
    data["language_instruction_timestamps"] = np.array([0], dtype=np.uint64)
    if embedding is not None:
        embedding = np.asarray(embedding, np.float32)
        if embedding.shape != (1, 512, 1024):
            raise ValueError(f"the instruction's embedding is {embedding.shape}, not 1 x 512 x 1024")
        data["language_embedding"] = embedding
        data["language_embedding_timestamps"] = np.array([0], dtype=np.uint64)
    return data, source


def read_frames(path: str) -> np.ndarray:
    """The pictures of a demonstration's frames.mp4 (steps x height x width x 3, RGB)."""
    import cv2

    video = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = video.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    video.release()
    return np.array(frames, np.uint8)


def episode_instruction(meta: dict) -> str:
    """The episode's one instruction; an episode with more than one stops the run."""
    instructions = set(meta["instructions"])
    if len(instructions) != 1:
        raise ValueError(f"{meta['episode_key']}: {len(instructions)} instructions in one episode")
    return instructions.pop()


def usable(meta: dict) -> Optional[str]:
    """None if the demonstration is a whole successful episode; else why it is left out."""
    if not meta["success"]:
        return "not successful"
    if meta["first_step"] != 0 or meta["steps"] != meta["episode_steps"]:
        return "not a whole episode"
    if meta["steps"] < 3:
        return "fewer than 3 steps"
    return None


def key_of(instruction: str) -> str:
    return "t5_" + hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:24]


def load_embeddings(path: str) -> Dict[str, np.ndarray]:
    """{instruction: embedding} of an embeddings file ({} if it is not there)."""
    if not os.path.exists(path):
        return {}
    with np.load(path) as f:
        instructions = json.loads(str(f["instructions"]))
        return {text: f[key_of(text)] for text in instructions}


def save_embeddings(path: str, embeddings: Dict[str, np.ndarray]) -> None:
    arrays = {key_of(text): np.asarray(e, np.float32) for text, e in embeddings.items()}
    np.savez(path, instructions=np.array(json.dumps(sorted(embeddings))), **arrays)


def t5_encoder(device: str, checkpoints: Optional[str] = None) -> Callable[[str], np.ndarray]:
    """mimic-video's own T5 text encoder (as its precompute_t5.py uses it): instruction -> 1 x 512 x 1024.
    `checkpoints`: mimic-video's checkpoints folder, if it is not the one of mimic-video's own setting."""
    if checkpoints:
        os.environ.setdefault("COSMOS_PREDICT2_ARGS", f"--checkpoints {shlex.quote(checkpoints)}")
    from imaginaire.auxiliary.text_encoder import CosmosT5TextEncoder, CosmosT5TextEncoderConfig
    from imaginaire.constants import T5_MODEL_DIR

    encoder = CosmosT5TextEncoder(config=CosmosT5TextEncoderConfig(ckpt_path=T5_MODEL_DIR), device=device)
    return lambda text: encoder.encode_prompts(text, max_length=512, return_mask=False).float().cpu().numpy()


def convert(folders: List[str], out: str, embeddings_path: str, limit: Optional[int] = None,
            make_encoder: Callable[[], Callable[[str], np.ndarray]] = lambda: t5_encoder("cpu"),
            frames_of: Callable[[str], np.ndarray] = read_frames, save: Optional[Callable] = None) -> dict:
    """Writes the usable demonstrations of `folders` (the first `limit` of them) to `out` and returns the report."""
    if save is None:
        import safetensors.numpy

        save = safetensors.numpy.save_file
    report = {"written": [], "left_out": {}, "gripper_state": {}, "instructions": {}}
    chosen = []
    for folder in folders:
        with open(os.path.join(folder, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        why = usable(meta)
        if why is not None:
            report["left_out"][why] = report["left_out"].get(why, 0) + 1
        elif limit is None or len(chosen) < limit:
            chosen.append((folder, meta, episode_instruction(meta)))
    embeddings = load_embeddings(embeddings_path)
    missing = sorted({text for _, _, text in chosen} - set(embeddings))
    if missing:
        encode = make_encoder()
        for text in missing:
            embeddings[text] = encode(text)
        save_embeddings(embeddings_path, embeddings)
    os.makedirs(out, exist_ok=True)
    for folder, meta, text in chosen:
        name = meta["episode_key"].replace("/", "__") + ".safetensors"
        with np.load(os.path.join(folder, "episode.npz")) as f:
            arrays = {k: f[k] for k in f.files if k in ("actions", "states", "fingers")}
        data, source = episode_tensors(frames_of(os.path.join(folder, "frames.mp4")), arrays, text, embeddings[text])
        save(data, os.path.join(out, name))
        report["written"].append(name)
        report["gripper_state"][source] = report["gripper_state"].get(source, 0) + 1
        report["instructions"][text] = report["instructions"].get(text, 0) + 1
    if len(set(report["written"])) != len(report["written"]):
        raise ValueError("two demonstrations have the same episode key")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("demos", nargs="+", help="folders with demonstrations (searched for meta.json)")
    parser.add_argument("--out", required=True, help="the folder for the episodes' safetensors files")
    parser.add_argument("--embeddings", required=True,
                        help="the instructions' T5 embeddings (an .npz file outside --out; made and added to)")
    parser.add_argument("--checkpoints", help="mimic-video's checkpoints folder (the T5 encoder is in its text_encoder)")
    parser.add_argument("--limit", type=int, help="take only the first N usable demonstrations")
    parser.add_argument("--t5-device", default="cpu", help="where mimic-video's T5 encoder runs (default: cpu)")
    args = parser.parse_args(argv)
    inside = os.path.commonpath([os.path.abspath(args.out), os.path.abspath(args.embeddings)])
    if inside == os.path.abspath(args.out):
        raise SystemExit("--embeddings must not be in --out: the training reads every file of --out as an episode")
    report = convert(find_episodes(args.demos), args.out, args.embeddings, args.limit,
                     make_encoder=lambda: t5_encoder(args.t5_device, args.checkpoints))
    report["written"] = len(report["written"])
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
