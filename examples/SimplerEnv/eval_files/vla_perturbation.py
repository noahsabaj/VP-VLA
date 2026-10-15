"""Input perturbation: how much the policy's first actions change when its input images change slightly.

The policy's action head is deterministic, so it gives no confidence of its own. Instead, on an episode's first
step the policy also runs on K copies of its input with a small camera-like change: the same random shift of all
the input images (up to MAX_SHIFT pixels, the edge pixels fill the gap) and the same brightness gain (1 +- MAX_GAIN).
The spread of the K predicted action chunks is a VLA-internal signal for the adaptive selector. The copies come
from a generator seeded with the episode key, so every run of an episode makes the same copies. Numpy only.
"""

import zlib
from typing import List

import numpy as np

MAX_SHIFT = 4  # pixels, on the policy's 224 x 224 input
MAX_GAIN = 0.1


def perturbation_rng(key: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(key.encode()))


def shift_image(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """The image moved dx pixels right and dy pixels down; the edge pixels fill the space left behind."""
    h, w = image.shape[:2]
    p = max(abs(dx), abs(dy))
    padded = np.pad(image, ((p, p), (p, p)) + ((0, 0),) * (image.ndim - 2), mode="edge")
    return padded[p - dy:p - dy + h, p - dx:p - dx + w]


def perturb_images(images: List[np.ndarray], rng: np.random.Generator, max_shift: int = MAX_SHIFT,
                   max_gain: float = MAX_GAIN) -> List[np.ndarray]:
    """One changed copy of the policy's input images (uint8): one shift and one gain for all of them."""
    dx, dy = (int(v) for v in rng.integers(-max_shift, max_shift + 1, size=2))
    gain = 1.0 + float(rng.uniform(-max_gain, max_gain))
    return [np.clip(np.rint(shift_image(im, dx, dy).astype(np.float32) * gain), 0, 255).astype(np.uint8)
            for im in images]


def action_spread(perturbed_actions: np.ndarray) -> float:
    """The signal: the standard deviation over the K copies of each normalized action value, averaged over the
    chunk's steps and the action dimensions. perturbed_actions: [K, chunk, action_dim]."""
    return float(np.std(np.asarray(perturbed_actions, dtype=np.float64), axis=0).mean())
