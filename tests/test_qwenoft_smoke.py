"""
CPU smoke test for the QwenOFT framework.

The real Qwen2.5-VL backbone is swapped for a tiny stub (character-level tokenizer + one embedding layer),
so this runs in seconds without GPUs, network access or pretrained weights. What it covers is the framework
glue around the VLM: prompt construction with action tokens, action-token gathering under left padding,
the L1 regression head, and the dtype/shape contract of `predict_action`.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image

import starVLA.model.framework.QwenOFT as qwen_oft
from starVLA.model.framework import build_framework

HIDDEN_SIZE = 32
ACTION_DIM = 7
FUTURE_WINDOW = 7
PAST_WINDOW = 0
CHUNK_LEN = PAST_WINDOW + 1 + FUTURE_WINDOW
VOCAB_SIZE = 256
ACTION_TOKEN = "🔍"
ACTION_TOKEN_ID = VOCAB_SIZE - 1
PAD_TOKEN_ID = 0


class _StubTokenizer:
    """Character-level tokenizer; the action token gets its own id."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self.encode_char(c) for c in text]}

    @staticmethod
    def encode_char(c):
        if c == ACTION_TOKEN:
            return ACTION_TOKEN_ID
        return 1 + ord(c) % (VOCAB_SIZE - 2)


class _StubBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=HIDDEN_SIZE)
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.image_proj = nn.Linear(3, HIDDEN_SIZE)
        self.layer = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)


class _StubVLInterface(nn.Module):
    """Mimics the parts of `_QWen_VL_Interface` that QwenOFT relies on."""

    def __init__(self):
        super().__init__()
        self.model = _StubBackbone()
        self.processor = SimpleNamespace(tokenizer=_StubTokenizer())

    def build_qwenvl_inputs(self, images, instructions, **kwargs):
        assert len(images) == len(instructions)
        token_lists = [self.processor.tokenizer(text)["input_ids"] for text in instructions]
        max_len = max(len(ids) for ids in token_lists)
        # left padding, same as the real processor
        input_ids = torch.full((len(token_lists), max_len), PAD_TOKEN_ID, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i, ids in enumerate(token_lists):
            input_ids[i, max_len - len(ids) :] = torch.tensor(ids)
            attention_mask[i, max_len - len(ids) :] = 1

        # mean RGB over every view of each sample, so the images actually condition the output
        pixel_values = []
        for views in images:
            assert all(isinstance(img, Image.Image) for img in views)
            pixel_values.append(np.stack([np.asarray(img, dtype=np.float32) / 255.0 for img in views]).mean((0, 1, 2)))
        pixel_values = torch.tensor(np.stack(pixel_values))

        weight = self.model.embed.weight
        return {
            "input_ids": input_ids.to(weight.device),
            "attention_mask": attention_mask.to(weight.device),
            "pixel_values": pixel_values.to(weight.device, weight.dtype),
        }

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, **kwargs):
        h0 = self.model.embed(input_ids) + self.model.image_proj(pixel_values).unsqueeze(1)
        h1 = torch.tanh(self.model.layer(h0)) * attention_mask.unsqueeze(-1).to(h0.dtype)
        return SimpleNamespace(hidden_states=(h0, h1))


def _make_config():
    return OmegaConf.create(
        {
            "framework": {
                "name": "QwenOFT",
                "qwenvl": {"base_vlm": "stub"},
                "action_model": {
                    "action_model_type": "MLP",
                    "action_dim": ACTION_DIM,
                    "future_action_window_size": FUTURE_WINDOW,
                    "past_action_window_size": PAST_WINDOW,
                },
            },
            "datasets": {"vla_data": {"image_size": [64, 64]}},
        }
    )


def _make_examples(batch_size, num_views=2):
    rng = np.random.default_rng(0)
    langs = ["pick up the red block", "open the drawer", "put the spoon in the cup"]
    return [
        {
            # raw uint8 HWC arrays, as the websocket policy server hands them over
            "image": [rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8) for _ in range(num_views)],
            "lang": langs[i % len(langs)],
            "action": rng.uniform(-1, 1, size=(CHUNK_LEN, ACTION_DIM)).astype(np.float32),
        }
        for i in range(batch_size)
    ]


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(qwen_oft, "get_vlm_model", lambda config: _StubVLInterface())
    torch.manual_seed(0)
    return build_framework(_make_config()).eval()


def test_builds_qwenoft_with_stub_vlm(model):
    assert isinstance(model, qwen_oft.Qwenvl_OFT)
    assert model.action_token_id == ACTION_TOKEN_ID
    assert model.chunk_len == CHUNK_LEN


@pytest.mark.parametrize("batch_size", [1, 3])
# bfloat16 weights stand in for GPU autocast: numpy has no bfloat16, so predict_action must upcast before returning
@pytest.mark.parametrize("model_dtype", [torch.float32, torch.bfloat16])
def test_predict_action_dtype_and_shape(model, batch_size, model_dtype):
    model.to(model_dtype)

    output = model.predict_action(examples=_make_examples(batch_size))

    actions = output["normalized_actions"]
    assert isinstance(actions, np.ndarray)
    assert actions.dtype == np.float32
    assert actions.shape == (batch_size, CHUNK_LEN, ACTION_DIM)
    assert np.isfinite(actions).all()


def test_forward_returns_finite_scalar_loss(model):
    model.train()
    examples = _make_examples(2)
    for example in examples:
        example["image"] = [Image.fromarray(img) for img in example["image"]]

    loss = model(examples)["action_loss"]

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert model.action_model.model.fc2.weight.grad is not None
