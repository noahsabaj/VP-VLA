"""CPU tests for the RGB camera noise (eval_files/camera_noise.py, numpy only)."""

import zlib

import numpy as np
import pytest

from examples.SimplerEnv.eval_files import camera_noise as cn

KEY = "PutSpoonOnTableClothInScene-v0/e12/3"


def _frames(noise, key, image, n=3):
    noise.new_episode(key)
    return [noise.apply(image) for _ in range(n)]


def test_no_noise_returns_the_same_picture():
    image = np.zeros((4, 4, 3), np.uint8)
    noise = cn.RGBNoise("none", 0)
    noise.new_episode(KEY)
    assert noise.apply(image) is image
    obs = {"image": {"3rd_view_camera": {"rgb": image}}}
    assert noise.apply_to_obs(obs, image) is image and obs["image"]["3rd_view_camera"]["rgb"] is image


def test_an_episode_depends_only_on_the_seed_and_the_key():
    image = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    a = _frames(cn.RGBNoise("harsh", 0), KEY, image)
    after_other = cn.RGBNoise("harsh", 0)
    _frames(after_other, "PutCarrotOnPlateInScene-v0/e12/3", image)
    b = _frames(after_other, KEY, image)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
        assert x.dtype == np.uint8 and x.shape == image.shape
    assert not np.array_equal(a[0], a[1])  # new sensor noise in every frame
    assert not np.array_equal(a[0], _frames(cn.RGBNoise("harsh", 1), KEY, image)[0])
    # its own stream: not the generator that DepthNoise seeds with (seed, key)
    depth_rng = np.random.default_rng([0, zlib.crc32(KEY.encode())])
    rgb_noise = cn.RGBNoise("harsh", 0)
    rgb_noise.new_episode(KEY)
    assert rgb_noise.rng.normal() != depth_rng.normal()


@pytest.mark.parametrize(("level", "sigma"), [("realistic", 4.2), ("harsh", 11.1)])
def test_sensor_noise_at_mid_gray(level, sigma):
    noise = cn.RGBNoise(level, 0)
    noise.new_episode(KEY)
    noise.gain = np.ones(3)  # no exposure or white-balance error, to see the sensor noise alone
    out = noise.apply(np.full((200, 200, 3), 128, np.uint8)).astype(np.float64)
    assert out.mean() == pytest.approx(128, abs=0.5)
    assert out.std() == pytest.approx(sigma, rel=0.1)


def test_episode_gains_are_fixed_for_the_episode():
    noise = cn.RGBNoise("harsh", 0)
    noise.new_episode(KEY)
    gain = noise.gain.copy()
    assert gain.shape == (3,) and not np.allclose(gain, 1.0)
    noise.apply(np.zeros((4, 4, 3), np.uint8))
    np.testing.assert_array_equal(noise.gain, gain)
    noise.new_episode("PutCarrotOnPlateInScene-v0/e12/3")
    assert not np.allclose(noise.gain, gain)


def test_gaussian_blur():
    flat = np.full((9, 9, 3), 0.5, np.float32)
    np.testing.assert_allclose(cn.gaussian_blur(flat, 1.5), flat, rtol=1e-6)
    impulse = np.zeros((41, 41), np.float32)
    impulse[20, 20] = 1.0
    out = cn.gaussian_blur(impulse, 2.0)
    assert out.sum() == pytest.approx(1.0, rel=1e-5)
    row = out.sum(axis=0)
    assert float((row * (np.arange(41) - 20) ** 2).sum()) == pytest.approx(4.0, rel=0.02)  # variance = sigma^2
    assert cn.gaussian_blur(impulse, 0) is impulse


def test_apply_to_obs_puts_the_noisy_picture_into_obs():
    image = np.full((8, 8, 3), 100, np.uint8)
    other = np.zeros((8, 8, 3), np.uint8)
    obs = {"image": {"3rd_view_camera": {"rgb": image, "depth": 1}, "base_camera": {"rgb": other}}}
    noise = cn.RGBNoise("realistic", 0)
    noise.new_episode(KEY)
    noisy = noise.apply_to_obs(obs, image)
    assert obs["image"]["3rd_view_camera"]["rgb"] is noisy and not np.array_equal(noisy, image)
    assert obs["image"]["base_camera"]["rgb"] is other and obs["image"]["3rd_view_camera"]["depth"] == 1
