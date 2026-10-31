"""CPU tests for the 1-GPU fine-tuning setup (train_files/run_finetune_1gpu.sh and the study's mixtures)."""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "examples/SimplerEnv/train_files/run_finetune_1gpu.sh"
STUDY = ("vla_p", "vla_pl", "vla_plt", "vla_s")
needs_bash = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None, reason="the launcher is a bash script"
)


def _freeze_list(frozen_layers):
    lm = "qwen_vl_interface.model.model.language_model"
    return [
        "qwen_vl_interface.model.model.visual",
        f"{lm}.embed_tokens",
        *(f"{lm}.layers.{i}" for i in range(frozen_layers)),
    ]


def _mixtures():
    spec = importlib.util.spec_from_file_location("mixtures", ROOT / "starVLA/dataloader/gr00t_lerobot/mixtures.py")
    mixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mixtures)
    return mixtures.DATASET_NAMED_MIXTURES


def test_each_model_of_the_study_has_its_own_set_in_the_bridge_format():
    sets = [_mixtures()[model] for model in STUDY]
    assert all(len(s) == 1 and s[0][2] == "oxe_bridge" for s in sets)  # the converter writes the Bridge keys
    assert len({s[0][0] for s in sets}) == len(STUDY)


def _setup(tmp_path, sets=("planner_demos",)):
    """A released run, a python for the launcher's checks and the sets of a mixture, all empty."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (bin_dir / "accelerate").write_text("#!/bin/sh\nexit 1\n")  # a dry run never starts it
    for name in ("python", "accelerate"):
        (bin_dir / name).chmod(0o755)
    run = tmp_path / "VP-VLA-OXE"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "steps_70000_pytorch_model.pt").write_bytes(b"")
    (run / "config.yaml").write_text("{}\n")
    for s in sets:
        (tmp_path / "data" / s / "meta").mkdir(parents=True)
        (tmp_path / "data" / s / "meta" / "info.json").write_text("{}")
        (tmp_path / "vp" / s).mkdir(parents=True)
    return {
        **os.environ,
        "STARVLA_PYTHON": str(bin_dir / "python"),
        "CKPT": str(run / "checkpoints" / "steps_70000_pytorch_model.pt"),
        "DATA_ROOT": str(tmp_path / "data"),
        "VP_DIR": str(tmp_path / "vp"),
        "RUN_ROOT": str(tmp_path / "runs"),
        "DRY_RUN": "1",
        "NUM_GPUS": "1",
    }


def _launch(env, *args, **settings):
    return subprocess.run(["bash", str(SCRIPT), *args], env={**env, **settings}, capture_output=True, text=True)


def _options(stdout):
    """The options of the printed command: {"--name": value}."""
    words = next(line for line in stdout.splitlines() if line.startswith("CUDA_VISIBLE_DEVICES=")).split()
    return {w: words[i + 1] for i, w in enumerate(words[:-1]) if w.startswith("--")}


@needs_bash
def test_the_dry_run_fine_tunes_the_released_checkpoint_on_one_gpu(tmp_path):
    env = _setup(tmp_path)
    result = _launch(env, "vla_p_1", "vla_p")
    assert result.returncode == 0, result.stderr
    assert "sets of vla_p: planner_demos" in result.stdout
    o = _options(result.stdout)
    assert o["--num_processes"] == "1" and o["--run_id"] == "vla_p_1"
    assert o["--config_yaml"] == str(tmp_path / "VP-VLA-OXE" / "config.yaml")  # every other setting is the release's
    assert o["--trainer.pretrained_checkpoint"] == env["CKPT"]
    assert o["--trainer.reload_modules"] == "qwen_vl_interface,action_model"  # strict key checks
    assert o["--use_inline_vp"] == "true"
    assert o["--datasets.vla_data.dataset_py"] == "visual_prompt_datasets_inline_vp"
    assert o["--datasets.vla_data.data_mix"] == "vla_p" and o["--datasets.vla_data.per_device_batch_size"] == "16"
    assert o["--trainer.freeze_modules"].split(",") == _freeze_list(24)
    assert o["--trainer.max_train_steps"] == o["--trainer.save_interval"] == "20000"
    assert not (tmp_path / "runs").exists()  # a dry run writes nothing

    o = _options(_launch(env, "vla_p_2", "vla_p", FROZEN_LAYERS="0", SAVE_INTERVAL="5000").stdout)
    assert o["--trainer.freeze_modules"].split(",") == _freeze_list(0)
    assert o["--trainer.save_interval"] == "5000"


@needs_bash
def test_the_launcher_stops_before_training_on_what_would_fail_later(tmp_path):
    env = _setup(tmp_path)
    (tmp_path / "runs" / "taken").mkdir(parents=True)
    for args, settings, error in [
        (("r", "no_such_mix"), {}, "no mixture no_such_mix"),
        (("r", "vla_pl"), {}, "no training set"),
        (("r", "vla_p"), {"MAX_STEPS": "100", "SAVE_INTERVAL": "30"}, "multiple of SAVE_INTERVAL"),
        (("r", "vla_p"), {"FROZEN_LAYERS": "37"}, "FROZEN_LAYERS must be"),
        (("taken", "vla_p"), {}, "exists"),
        (("r", "vla_p"), {"CKPT": str(tmp_path / "none.pt")}, "no checkpoint"),
    ]:
        result = _launch(env, *args, **settings)
        assert result.returncode != 0 and error in result.stderr, (args, settings, result.stderr)


def test_the_upper_layers_and_the_action_head_train(capsys):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")  # trainer_tools imports it
    transformers = pytest.importorskip("transformers")
    from omegaconf import OmegaConf

    from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups

    # A tiny Qwen3-VL with Qwen3-VL-4B's 36 layers and tied word embeddings, in QwenOFT's two parts.
    text = {"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 36, "num_attention_heads": 2}
    text.update(num_key_value_heads=1, head_dim=8, vocab_size=64, tie_word_embeddings=True)
    text.update(rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True})
    vision = {"depth": 2, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2, "out_hidden_size": 16}
    vision.update(deepstack_visual_indexes=[1])
    config = transformers.Qwen3VLConfig(text_config=text, vision_config=vision, tie_word_embeddings=True)
    vlm = transformers.Qwen3VLForConditionalGeneration(config)
    model = torch.nn.Module()
    model.qwen_vl_interface = torch.nn.Module()
    model.qwen_vl_interface.model = vlm
    model.action_model = torch.nn.Linear(16, 7)
    lm = vlm.model.language_model
    assert vlm.lm_head.weight is lm.embed_tokens.weight

    freeze = ",".join(_freeze_list(24))
    TrainerUtils.freeze_backbones(model, freeze_modules=freeze)
    assert "does not exist" not in capsys.readouterr().out  # every path names a module
    lr = {"base": 3e-5, "qwen_vl_interface": 1e-5, "action_model": 1e-4}
    cfg = OmegaConf.create({"trainer": {"freeze_modules": freeze, "learning_rate": lr}})
    groups = build_param_lr_groups(model, cfg)

    trains = {id(p) for p in (*lm.layers[24:].parameters(), *lm.norm.parameters(), *model.action_model.parameters())}
    assert {id(p) for g in groups for p in g["params"]} == trains  # the optimizer's
    assert {id(p) for p in model.parameters() if p.requires_grad} == trains
    assert {g["name"]: g["lr"] for g in groups} == {"qwen_vl_interface": 1e-5, "action_model": 1e-4}
