"""CPU tests for the VLA's input perturbation (eval_files/vla_perturbation.py, numpy only)."""

import numpy as np
import pytest

from examples.SimplerEnv.eval_files import vla_perturbation as vp


def test_shift_image_moves_the_pixels_and_fills_from_the_edge():
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)
    right = vp.shift_image(image, 1, 0)
    np.testing.assert_array_equal(right[:, 1:], image[:, :-1])
    np.testing.assert_array_equal(right[:, 0], image[:, 0])
    up = vp.shift_image(image, 0, -2)
    np.testing.assert_array_equal(up[:2], image[2:])
    np.testing.assert_array_equal(up[2:], np.stack([image[3], image[3]]))
    np.testing.assert_array_equal(vp.shift_image(image, 0, 0), image)
    assert vp.shift_image(np.zeros((4, 4, 3), np.uint8), 2, 2).shape == (4, 4, 3)


def test_perturb_images_repeats_for_the_same_key_and_changes_all_images_alike():
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 256, (8, 8, 3), dtype=np.uint8) for _ in range(2)]
    key = "PutSpoonOnTableClothInScene-v0/e12/3"
    a = vp.perturb_images(images, vp.perturbation_rng(key))
    b = vp.perturb_images(images, vp.perturbation_rng(key))
    for x, y, original in zip(a, b, images, strict=True):
        np.testing.assert_array_equal(x, y)
        assert x.dtype == np.uint8 and x.shape == original.shape
        assert not np.array_equal(x, original)
    twice = vp.perturb_images([images[0], images[0]], vp.perturbation_rng(key))
    np.testing.assert_array_equal(twice[0], twice[1])  # one shift and one gain for all the images
    unchanged = vp.perturb_images(images, rng, max_shift=0, max_gain=0.0)
    np.testing.assert_array_equal(unchanged[0], images[0])


def test_action_spread():
    assert vp.action_spread(np.ones((8, 4, 7))) == 0.0
    two = np.stack([np.zeros((4, 7)), np.full((4, 7), 2.0)])
    assert vp.action_spread(two) == pytest.approx(1.0)
