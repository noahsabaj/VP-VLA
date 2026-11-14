"""The frontier-model study's plan arm, on the simulator's side (experiments/README.md, "Frontier-model study").

The frontier model answers through an API, and the shared GPU server holds no API key. So the plan arm runs in three
parts:

1. `--controller frames` (FrameExporter) saves each episode's first frame, the picture that the VLM plan gets, with a
   line in frames.jsonl: the episode, its instruction and the picture's hash. The robot does not move.
2. On a machine that holds the key, experiments/frontier_answers.py asks the model the VLM plan's question
   (vlm_plan.prompt) about each frame, once, and saves the answers.
3. `--task-source vlm --vlm-plan-answers FILE` (SavedAnswers): the planner-only eval reads the saved answers in place
   of the VLM server. The parser and the planner after it are the VLM plan's, unchanged.

An answer is tied to its episode, its question and its picture: the eval stops when one of them is not the saved
one, so an answer can never be used for another episode, another wording or another frame.

Needs numpy, and Pillow or OpenCV for the PNG files.
"""

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import vlm_plan

MODEL = "claude-mythos-5-2"  # the study's frontier model, for both arms; there is no fallback model
EFFORT = "high"
# USD per million tokens (the model's list prices on 2026-11-14), for the cost that is reported: input, output, a
# cache write (5 minutes) and a cache read.
PRICES = {"claude-mythos-5-2": {"input": 8.0, "output": 40.0, "cache_write": 10.0, "cache_read": 0.2}}
USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
FRAMES_FILE = "frames.jsonl"  # in a task's run folder, next to its frames/ folder
# The agent arm's record of each episode, one line in a task's run folder: "obj_episode_id", "success", "turns" (the
# model's turns), "end" (one of AGENT_ENDS: how the model's part ended; a refusal or an error ends it as give_up,
# with "refusal": true or the reason in "error"), "usage" (the tokens of all its requests), "cost_usd" (at the list
# price) and "model_seconds" (the time of its requests). experiments/score_frontier.py reads it.
AGENT_FILE = "agent_stats.jsonl"
AGENT_ENDS = ("done", "give_up", "turn limit", "step limit")


def image_hash(image) -> str:
    """A picture's identity: the SHA-256 of its shape and its uint8 RGB bytes."""
    a = np.ascontiguousarray(image, dtype=np.uint8)
    return hashlib.sha256(repr(tuple(a.shape)).encode() + a.tobytes()).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def usage_cost(usage: dict, model: Optional[str] = MODEL) -> Optional[float]:
    """The list-price cost in USD of one usage record (USAGE_FIELDS); None when the model has no price here."""
    price = PRICES.get(model)
    if price is None:
        return None
    return ((usage.get("input_tokens") or 0) * price["input"] + (usage.get("output_tokens") or 0) * price["output"]
            + (usage.get("cache_creation_input_tokens") or 0) * price["cache_write"]
            + (usage.get("cache_read_input_tokens") or 0) * price["cache_read"]) / 1e6


def _pil():
    """Pillow's Image module, or None where it is not installed (the simulator's environment has OpenCV)."""
    try:
        from PIL import Image
    except ImportError:
        return None
    return Image


def write_png(path: str, rgb: np.ndarray) -> None:
    """An RGB picture as a PNG file (lossless, so the file gives back the same pixels)."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    pil = _pil()
    if pil is not None:
        pil.fromarray(rgb).save(path, format="PNG")
        return
    import cv2

    if not cv2.imwrite(path, rgb[..., ::-1]):
        raise OSError(f"could not write {path}")


def png_bytes(rgb: np.ndarray) -> bytes:
    """An RGB picture as the bytes of a PNG file."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    pil = _pil()
    if pil is not None:
        import io

        buffer = io.BytesIO()
        pil.fromarray(rgb).save(buffer, format="PNG")
        return buffer.getvalue()
    import cv2

    ok, data = cv2.imencode(".png", rgb[..., ::-1])
    if not ok:
        raise OSError("could not encode the picture")
    return data.tobytes()


def read_png(path: str) -> np.ndarray:
    """A PNG file's picture as uint8 RGB."""
    pil = _pil()
    if pil is not None:
        with pil.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    import cv2

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"could not read {path}")
    return np.ascontiguousarray(bgr[..., ::-1])


class FrameExporter:
    """A controller that saves each episode's first frame and ends the episode (the evaluator's model interface).

    In `out_dir` (a task's run folder): frames/KEY.png, and one line per episode in frames.jsonl with the episode's
    key (planner_evaluator's: task/layout set/episode), its instruction, the file and the picture's hash."""

    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir
        self.key: Optional[str] = None
        self.task_description = ""
        self.saved = True
        self.seen = set()

    def set_episode(self, key: str) -> None:
        self.key = key

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.saved = False

    def step(self, image, task_description: Optional[str] = None, **kwargs):
        if not self.saved:
            self._save(image, self.task_description if task_description is None else task_description)
            self.saved = True
        zero = np.zeros(3)
        raw_action = {"world_vector": zero, "rotation_delta": zero, "open_gripper": np.ones(1)}
        action = {"world_vector": zero, "rot_axangle": zero, "gripper": np.ones(1), "terminate_episode": np.ones(1)}
        return raw_action, action

    def _save(self, image, instruction: str) -> None:
        if self.key is None:
            raise ValueError("the frame export needs the episode key (planner_evaluator sets it)")
        if self.key in self.seen:
            raise ValueError(f"a second first frame for episode {self.key}")
        self.seen.add(self.key)
        image = np.ascontiguousarray(image, dtype=np.uint8)
        name = os.path.join("frames", self.key.replace("/", "__") + ".png")
        os.makedirs(os.path.join(self.out_dir, "frames"), exist_ok=True)
        write_png(os.path.join(self.out_dir, name), image)
        row = {"key": self.key, "instruction": instruction, "frame": name.replace(os.sep, "/"),
               "image_sha256": image_hash(image), "width": int(image.shape[1]), "height": int(image.shape[0])}
        with open(os.path.join(self.out_dir, FRAMES_FILE), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def visualize_epoch(self, predicted_raw_actions, images, save_path: str) -> None:
        pass


def read_frames(root: str) -> List[dict]:
    """Every first frame under `root` (a run folder of frames_eval.sh, or one task's folder): the rows of its
    frames.jsonl files, in file and line order, each with "path" (its PNG file). A second row for an episode and its
    instruction, or a row without its file, stops the load."""
    rows, seen = [], set()
    files = sorted(os.path.join(d, FRAMES_FILE) for d, _, names in os.walk(root) if FRAMES_FILE in names)
    if not files:
        raise SystemExit(f"{root}: no {FRAMES_FILE} (frames_eval.sh writes one in each task's folder)")
    for file in files:
        with open(file, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                k = (row["key"], row["instruction"])
                if k in seen:
                    raise SystemExit(f"{file}:{n}: a second frame for episode {k[0]} with the instruction {k[1]!r}")
                seen.add(k)
                row["path"] = os.path.join(os.path.dirname(file), *row["frame"].split("/"))
                if not os.path.isfile(row["path"]):
                    raise SystemExit(f"{file}:{n}: no file {row['path']}")
                rows.append(row)
    return rows


def read_answers(path: str) -> Dict[Tuple[str, str], dict]:
    """The saved answers of an answers file (experiments/frontier_answers.py), by (episode key, instruction). A second
    answer for an episode stops the load: every episode is asked once."""
    rows: Dict[Tuple[str, str], dict] = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            k = (row["key"], row["instruction"])
            if k in rows:
                raise SystemExit(f"{path}:{n}: a second answer for episode {k[0]} with the instruction {k[1]!r}; "
                                 "every episode is asked once")
            rows[k] = row
    return rows


class SavedAnswers:
    """Stands in for the VLM server of the VLM plan: each episode's saved answer to vlm_plan.prompt."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.rows = read_answers(path)
        if not self.rows:
            raise SystemExit(f"{path}: no saved answers")

    def answer(self, key: Optional[str], instruction: str, image) -> dict:
        """{"text", "model", "error"} of the episode's saved answer. A refusal or an error has an empty text, which
        the VLM plan's parser reads as not feasible. The eval stops when the episode has no saved answer, or when its
        question or its first frame is not the one that the model answered."""
        row = self.rows.get((key, instruction))
        if row is None:
            raise SystemExit(f"{self.path}: no saved answer for episode {key} with the instruction {instruction!r}")
        if row["prompt_sha256"] != text_hash(vlm_plan.prompt(instruction)):
            raise SystemExit(f"{self.path}: episode {key}: the VLM plan's question is not the one that was answered "
                             "(vlm_plan.prompt has changed since)")
        if row["image_sha256"] != image_hash(image):
            raise SystemExit(f"{self.path}: episode {key}: this eval's first frame is not the picture that the model "
                             "answered")
        return {"text": row.get("text") or "", "model": row.get("model"), "error": row.get("error")}
