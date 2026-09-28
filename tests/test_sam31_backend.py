"""
Contract test for the SAM 3.1 backend's output conversion.

Meta's `sam3` package and the 3.5 GB checkpoint are not installed in CI, so this only checks that
`state_to_result` turns a Sam3Processor state into the same reply format the transformers SAM3 server sends:
(N, H, W) uint8 masks thresholded at `mask_threshold`, (N, 4) float32 xyxy boxes and (N,) float32 scores.
"""

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
