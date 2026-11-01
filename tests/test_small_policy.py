"""CPU tests for the marriage study's small policy (train_files/small_policy.py)."""

import itertools
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from examples.SimplerEnv.eval_files.instruction_sets import OWN, TASKS  # noqa: E402
from examples.SimplerEnv.train_files import small_policy as sp  # noqa: E402

# what VP-VLA's client puts before the instruction (model2simpler_interface_visual_prompt.py)
PREFIX = (
    "You are given two images: the first is the original robot observation, and the second has visual prompts "
    "overlaid highlighting the target object and target location. "
)


def constant_policy(value):
    """A policy whose every action number is `value`."""
    model = sp.SmallPolicy(pretrained=False, hidden=8)
    torch.nn.init.zeros_(model.head[-1].weight)
    torch.nn.init.constant_(model.head[-1].bias, value)
    return model.eval()


def test_the_server_picks_each_tasks_policy():
    assert [sp.task_of(PREFIX + own) for own in OWN] == list(TASKS)
    policies = sp.TaskPolicies({task: constant_policy(i) for i, task in enumerate(TASKS)})
    picture = np.zeros((32, 32, 3), np.uint8)
    examples = [{"image": [picture, picture], "lang": PREFIX + OWN[i]} for i in (3, 1)]
    # the arguments of VP-VLA's head do not apply
    chunks = policies.predict_action(examples=examples, do_sample=False, use_ddim=True, num_ddim_steps=10)
    chunks = chunks["normalized_actions"]
    assert chunks.shape == (2, 16, 7) and chunks.dtype == np.float32
    assert np.allclose(chunks[0], 3) and np.allclose(chunks[1], 1)
    with pytest.raises(ValueError):
        sp.task_of(PREFIX + "open the top drawer")


def test_a_checkpoint_loads_the_same_policies(tmp_path):
    torch.manual_seed(0)
    models = {task: sp.SmallPolicy(pretrained=False, hidden=16).eval() for task in TASKS}
    path = tmp_path / "steps_1_pytorch_model.pt"
    sp.save(path, models, {"hidden": 16})
    loaded = sp.load(path)
    images = torch.randint(0, 256, (1, 2, 32, 32, 3), dtype=torch.uint8)
    with torch.no_grad():
        for task in TASKS:
            assert torch.allclose(loaded.models[task](images), models[task](images))


def test_random_shift_moves_both_pictures_alike():
    images = torch.zeros((5, 2, 20, 20, 3), dtype=torch.uint8)
    images[:, :, 10, 10] = 255  # one bright pixel in the middle of both pictures
    assert sp.random_shift(images, 0) is images
    shifted = sp.random_shift(images, 4, torch.Generator().manual_seed(0))
    assert shifted.shape == images.shape
    for sample in shifted:
        assert torch.equal(sample[0], sample[1])
        (y, x), *rest = (sample[0, :, :, 0] == 255).nonzero().tolist()
        assert not rest and abs(y - 10) <= 4 and abs(x - 10) <= 4


def test_batch_arrays():
    from PIL import Image

    picture = Image.fromarray(np.full((224, 224, 3), 7, np.uint8))
    images, actions = sp.batch_arrays([{"image": [picture, picture], "action": np.ones((16, 7), np.float16)}] * 2)
    assert images.shape == (2, 2, 224, 224, 3) and images.dtype == np.uint8 and images.max() == 7
    assert actions.shape == (2, 16, 7) and actions.dtype == np.float32
    with pytest.raises(ValueError):  # the camera picture only
        sp.batch_arrays([{"image": [picture], "action": np.ones((16, 7))}])
    with pytest.raises(ValueError):
        sp.batch_arrays([{"image": [picture, picture], "action": np.ones((8, 7))}])


def test_episode_tasks(tmp_path):
    rows = [
        {"episode_index": 0, "episode_key": "PutCarrotOnPlateInScene-v0/iid100/3"},
        {"episode_index": 1, "episode_key": "TwoStepSpoonAndCarrot-v0/iid100/0"},
    ]
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "sources.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert sp.episode_tasks(tmp_path) == {0: "PutCarrotOnPlateInScene-v0", 1: "TwoStepSpoonAndCarrot-v0"}


def test_the_learning_rate_warms_up_then_decays():
    factors = [sp.lr_factor(s, 100, 10) for s in range(100)]
    assert factors[0] == pytest.approx(0.1) and factors[9] == factors[10] == 1
    assert all(a >= b for a, b in itertools.pairwise(factors[10:])) and factors[-1] < 1e-3
