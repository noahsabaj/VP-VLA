"""
Contract tests for the SAM 3.1 backend: its output conversion and its image cache.

Meta's `sam3` package and the 3.5 GB checkpoint are not installed in CI, so these tests check, without the model,
that `state_to_result` turns a Sam3Processor state into the same reply format the transformers SAM3 server sends
((N, H, W) uint8 masks thresholded at `mask_threshold`, (N, 4) float32 xyxy boxes and (N,) float32 scores), and,
with a stand-in processor, that SAM31Model runs the image encoder once per frame.
"""

import collections
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "Robocasa_tabletop" / "visual_prompt_utility"))

from sam31_backend import state_to_result


def _state(probs, boxes, scores):
    return {
        "masks_logits": torch.tensor(probs, dtype=torch.float32).unsqueeze(1),
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "scores": torch.tensor(scores, dtype=torch.float32),
    }


def test_state_to_result_matches_server_format():
    probs = np.zeros((2, 4, 5), dtype=np.float32)
    probs[0, 1:3, 1:4] = 0.9
    probs[1, 0, 0] = 0.4
    state = _state(probs, [[1.0, 1.0, 4.0, 3.0], [0.0, 0.0, 1.0, 1.0]], [0.8, 0.6])

    result = state_to_result(state, mask_threshold=0.5)

    assert result["num_masks"] == 2
    assert result["masks"].shape == (2, 4, 5)
    assert result["masks"].dtype == np.uint8
    assert result["masks"][0].sum() == 6
    assert result["masks"][1].sum() == 0
    assert result["boxes"].dtype == np.float32
    assert result["boxes"].shape == (2, 4)
    np.testing.assert_allclose(result["scores"], [0.8, 0.6])


def test_mask_threshold_is_applied():
    probs = np.full((1, 2, 2), 0.4, dtype=np.float32)
    state = _state(probs, [[0.0, 0.0, 2.0, 2.0]], [0.9])

    assert state_to_result(state, mask_threshold=0.5)["masks"].sum() == 0
    assert state_to_result(state, mask_threshold=0.3)["masks"].sum() == 4


def test_no_detections():
    state = _state(np.zeros((0, 3, 3), dtype=np.float32), np.zeros((0, 4)), np.zeros((0,)))

    result = state_to_result(state, mask_threshold=0.5)

    assert result["num_masks"] == 0
    assert result["masks"].shape == (0, 3, 3)
    assert result["boxes"].shape == (0, 4)


class _FakeProcessor:
    """Sam3Processor stand-in: counts image-encoder passes; each prompt finds one box as wide as its name."""

    def __init__(self):
        self.images = 0
        self.confidence_threshold = None

    def set_image(self, image):
        self.images += 1
        return {"backbone_out": {"image": np.asarray(image)}}

    def reset_all_prompts(self, state):
        for key in ("prompt", "masks_logits", "boxes", "scores"):
            state.pop(key, None)

    def set_text_prompt(self, prompt, state):
        assert "prompt" not in state, "the last prompt's results must be cleared first"
        h, w = state["backbone_out"]["image"].shape[:2]
        probs = np.zeros((1, h, w), dtype=np.float32)
        probs[0, 0, : len(prompt)] = 0.9
        state.update(_state(probs, [[0.0, 0.0, float(len(prompt)), 1.0]], [0.7]), prompt=prompt)
        return state


def _model():
    from sam31_backend import SAM31Model

    model = SAM31Model.__new__(SAM31Model)  # no checkpoint: only the image cache and the prompt loop
    model.processor = _FakeProcessor()
    model._states = collections.OrderedDict()
    return model


def test_image_encoder_runs_once_per_frame():
    model = _model()
    frame = np.zeros((4, 8, 3), dtype=np.uint8)

    results = model.segment_many(frame, ["cup", "basket"], threshold=0.3)
    again = model.segment(frame, "cup", threshold=0.1)  # e.g. the planner's retry at a lower threshold

    assert model.processor.images == 1
    assert model.processor.confidence_threshold == 0.1
    assert [r["masks"][0].sum() for r in results] == [3, 6]
    np.testing.assert_array_equal(again["masks"], results[0]["masks"])
    model.segment(frame + 1, "cup")
    assert model.processor.images == 2
