"""RGB camera noise: a real color camera's errors in the picture that every controller gets.

The simulator's picture is perfect. A real camera adds:
* per episode (one camera setup, fixed for the whole episode): an exposure error (every pixel brighter or darker by
  one gain), a white-balance error (one gain per color channel) and a blur (defocus: a Gaussian blur of fixed width);
* per frame: sensor noise, Poisson-Gaussian (shot noise that grows with the brightness, plus read noise; Foi et al.,
  2008): sigma^2 = shot * I + read^2 for a pixel value I in [0, 1].

The evaluator (planner/planner_evaluator.py) applies it once per frame, before any controller sees the picture, so
the VLA (its input, its SAM3 visual prompts and its VLM) and the planner (its SAM3 masks) see the same noisy pixels.
Depth is not changed here: the planner's depth errors are DepthNoise (planner/geometry.py).

With a seed and an episode key, an episode's draws depend only on (seed, key) and the frame number, so splitting or
reordering a run gives the same noise, and every controller gets the same first frame. The draws come from a stream
of their own, so turning the RGB noise on does not change the depth noise's draws. Numpy only.
"""

import zlib
from typing import Optional

import numpy as np

STREAM = 1  # DepthNoise seeds its generator with (seed, key); this one with (seed, key, STREAM)


class RGBNoise:
    """The levels: shot and read noise (pixel values in [0, 1]), exposure and white_balance (standard deviations of
    the log gain: 0.05 is about +-5%), blur_px (the Gaussian blur's sigma in pixels, on the 640 x 480 picture)."""

    LEVELS = {
        "none": None,
        # A good camera in lab light: noise of about 4/255 at mid-gray, exposure +-5%, slight blur.
        "realistic": dict(shot=0.0005, read=0.005, exposure=0.05, white_balance=0.02, blur_px=0.5),
        # A cheap camera in dim light: noise of about 11/255 at mid-gray, exposure +-16% (a quarter stop), soft focus.
        "harsh": dict(shot=0.003, read=0.02, exposure=0.15, white_balance=0.05, blur_px=1.5),
    }

    def __init__(self, level: str = "none", seed: Optional[int] = None):
        self.level = level
        self.cfg = self.LEVELS[level]
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.gain = np.ones(3)

    @property
    def active(self) -> bool:
        return self.cfg is not None

    def new_episode(self, key: Optional[str] = None) -> None:
        """Draw this episode's exposure and white-balance gains. `key` names the episode (task, layout seed, episode
        id); without it, the draws go on from the previous episode's."""
        if self.seed is not None and key is not None:
            self.rng = np.random.default_rng([self.seed, zlib.crc32(key.encode()), STREAM])
        self.gain = np.ones(3)
        if not self.active:
            return
        c = self.cfg
        self.gain = np.exp(c["exposure"] * self.rng.normal() + c["white_balance"] * self.rng.normal(size=3))

    def apply(self, image: np.ndarray) -> np.ndarray:
        """One frame (H, W, 3) uint8 through the camera: blur, gains, sensor noise. Without noise, the same array."""
        if not self.active:
            return image
        c = self.cfg
        x = gaussian_blur(image.astype(np.float32) / 255.0, c["blur_px"]) * self.gain.astype(np.float32)
        sigma = np.sqrt(c["shot"] * np.clip(x, 0.0, 1.0) + c["read"] ** 2)
        x = x + sigma * self.rng.standard_normal(x.shape, dtype=np.float32)
        return np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)

    def apply_to_obs(self, obs: dict, image: np.ndarray) -> np.ndarray:
        """apply() for `image`, one camera's picture in the observation obs. The noisy picture also replaces it in
        obs, so whatever reads obs (the planner's SAM3, cuRobo) sees the same pixels as the VLA."""
        if not self.active:
            return image
        noisy = self.apply(image)
        for camera in obs["image"].values():
            if camera.get("rgb") is image:
                camera["rgb"] = noisy
        return noisy


def gaussian_blur(x: np.ndarray, sigma: float) -> np.ndarray:
    """x (H, W, ...) blurred over its first two axes with a Gaussian of `sigma` pixels (the edge pixels fill the
    border)."""
    if sigma <= 0:
        return x
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k = (k / k.sum()).astype(x.dtype)
    for axis in (0, 1):
        pad = [(0, 0)] * x.ndim
        pad[axis] = (r, r)
        p = np.pad(x, pad, mode="edge")
        n = x.shape[axis]
        out = np.zeros_like(x)
        for i, w in enumerate(k):
            out += w * p[(slice(None),) * axis + (slice(i, i + n),)]
        x = out
    return x
