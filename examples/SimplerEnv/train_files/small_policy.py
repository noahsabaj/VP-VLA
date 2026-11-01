"""The marriage study's small policy (experiments README, "Marriage study"): what the VLA adds beyond the
demonstrations. One policy for each of SimplerEnv's 4 WidowX tasks, trained from scratch on VLA-P's demonstrations of
that task: a ResNet-18 image encoder with ImageNet weights, shared by the camera picture and the picture with the
visual prompts, and an MLP that gives the next 16 steps' 7 action numbers, as VP-VLA's action head does, trained with
an L1 loss. It reads no language.

    python -m examples.SimplerEnv.train_files.small_policy train --config CHECKPOINT_RUN/config.yaml \
        --data-root DATA_ROOT --visual-prompts VP_DIR --out RUN_ROOT/small
    python -m examples.SimplerEnv.train_files.small_policy serve --ckpt CKPT --port PORT

train makes its samples with VP-VLA's own training data code (starVLA/dataloader/visual_prompt_datasets.py, with the
data settings of the released run's config and the mixture vla_p), so each sample is one that VLA-P's training draws:
the same two pictures, the same visual prompts and the same scaled action chunk. It makes every sample of the task's
episodes once (meta/sources.jsonl names each episode's task), keeps them on the GPU and draws the batches from them.
OUT gets the layout that the evals read: checkpoints/steps_N_pytorch_model.pt (the 4 tasks' policies), config.yaml
(the settings), dataset_statistics.json (the actions' scale, as VLA-P's run has it), train_log.jsonl and summary.json.

serve answers VP-VLA's eval client (eval_files/model2simpler_interface_visual_prompt.py) as VP-VLA's policy server
does (deployment/model_server/server_policy.py), so the eval is VLA-P's but for the policy: the same visual prompts,
the same action chunks' ensembling and the same scale. It picks each request's policy by the task's own instruction,
which the client sends; the runners start it with POLICY=small (common.sh).

Needs torch and torchvision; train also needs the starVLA environment and serve the policy server's websockets.
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from examples.SimplerEnv.eval_files.instruction_sets import OWN, TASKS

CHUNK, DIMS = 16, 7  # VP-VLA's action chunk: the next 16 steps' 7 action numbers
PICTURES = 2  # the camera picture and the picture with the visual prompts
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class SmallPolicy(nn.Module):
    """One task's policy: the ResNet-18 features of its two pictures, then an MLP to the scaled action chunk."""

    def __init__(self, pretrained: bool = True, hidden: int = 512):
        super().__init__()
        import torchvision

        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.encoder = torchvision.models.resnet18(weights=weights)
        self.encoder.fc = nn.Identity()  # the 512 pooled features
        self.head = nn.Sequential(nn.Linear(PICTURES * 512, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, CHUNK * DIMS))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, 2, H, W, 3] RGB, 0 to 255; returns the scaled actions [B, 16, 7]."""
        b = images.shape[0]
        x = images.flatten(0, 1).permute(0, 3, 1, 2).float() / 255
        features = self.encoder((x - self.mean) / self.std).reshape(b, -1)
        return self.head(features).view(b, CHUNK, DIMS)


def task_of(lang: str) -> str:
    """The task whose own instruction ends the text that VP-VLA's client sends (its sentence on the pictures comes
    first)."""
    found = [task for task, own in zip(TASKS, OWN, strict=True) if lang.endswith(own)]
    if len(found) != 1:
        raise ValueError(f"the small policies know only the 4 tasks' own instructions, not {lang!r}")
    return found[0]


class TaskPolicies:
    """The policy that serve gives VP-VLA's policy server: each example goes to its task's small policy."""

    def __init__(self, models: Dict[str, SmallPolicy], device: str = "cpu"):
        self.models, self.device = models, device

    @torch.no_grad()
    def predict_action(self, examples: List[dict], **_) -> dict:
        """VP-VLA's predict_action: examples [{"image": [camera picture, picture with the visual prompts], "lang"}]
        -> {"normalized_actions": [B, 16, 7]}. The other arguments (the sampling of VP-VLA's head) do not apply."""
        chunks = []
        for example in examples:
            images = torch.from_numpy(np.stack([np.asarray(im, dtype=np.uint8) for im in example["image"]]))
            model = self.models[task_of(example["lang"])]
            chunks.append(model(images[None].to(self.device))[0].float().cpu().numpy())
        return {"normalized_actions": np.stack(chunks)}


def save(path, models: Dict[str, nn.Module], settings: dict) -> None:
    states = {task: {k: v.cpu() for k, v in model.state_dict().items()} for task, model in models.items()}
    torch.save({"tasks": states, "settings": settings}, path)


def load(path, device: str = "cpu") -> TaskPolicies:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    models = {}
    for task, state in checkpoint["tasks"].items():
        model = SmallPolicy(pretrained=False, hidden=checkpoint["settings"]["hidden"])
        model.load_state_dict(state)
        models[task] = model.to(device).eval()
    return TaskPolicies(models, device)


def random_shift(images: torch.Tensor, pad: int, generator=None) -> torch.Tensor:
    """Each sample's two pictures [B, 2, H, W, 3] moved by the same random amount, up to `pad` pixels each way, with the
    edge pixels repeated: the training's only augmentation (the evals take the pictures as they are)."""
    if pad == 0:
        return images
    b, n, h, w, c = images.shape
    x = images.permute(0, 1, 4, 2, 3).reshape(b, n * c, h, w).float()
    x = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    device = generator.device if generator is not None else images.device
    dx, dy = torch.randint(0, 2 * pad + 1, (2, b), generator=generator, device=device)
    x = torch.stack([x[i, :, y:y + h, z:z + w] for i, (y, z) in enumerate(zip(dy.tolist(), dx.tolist(), strict=True))])
    return x.reshape(b, n, c, h, w).permute(0, 1, 3, 4, 2)


def batch_arrays(samples: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """The pictures [B, 2, H, W, 3] uint8 and the scaled action chunks [B, 16, 7] float32 of VP-VLA's training samples
    ("image": the camera picture and the picture with the visual prompts, "action")."""
    for sample in samples:
        if len(sample["image"]) != PICTURES:
            raise ValueError(f"a sample with {len(sample['image'])} pictures, not 2 (the config's feed_both_images)")
    images = np.stack([np.stack([np.asarray(im, dtype=np.uint8) for im in s["image"]]) for s in samples])
    actions = np.stack([np.asarray(s["action"], dtype=np.float32) for s in samples])
    if actions.shape[1:] != (CHUNK, DIMS):
        raise ValueError(f"action chunks of {actions.shape[1:]}, not {(CHUNK, DIMS)}")
    return images, actions


def episode_tasks(dataset_path) -> Dict[int, str]:
    """Each episode's task (its environment) in a set that planner_demos_to_lerobot.py wrote (meta/sources.jsonl)."""
    with open(Path(dataset_path) / "meta" / "sources.jsonl") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return {row["episode_index"]: str(row["episode_key"] or "").split("/")[0] for row in rows}


def lr_factor(step: int, steps: int, warmup: int) -> float:
    """The learning rate's factor at a step: a linear warm-up, then a cosine decay to 0."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))


def task_samples(mixture, task: str, batch: int, workers: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Every sample of the task's episodes in the mixture's sets, once each and as VP-VLA's training makes it (its
    __getitem__, with the step given instead of drawn): the pictures, the action chunks and the number of episodes."""
    from torch.utils.data import DataLoader

    from starVLA.dataloader.visual_prompt_datasets import collate_fn

    steps, episodes = [], 0
    for single in mixture.datasets:
        tasks = episode_tasks(single.dataset_path)
        mine = [(single, int(t), int(s)) for t, s in single.all_steps if tasks[int(t)] == task]
        steps += mine
        episodes += len({t for _, t, _ in mine})
    if not steps:
        raise SystemExit(f"no episodes of {task} in the sets of the mixture")
    mixture.sample_step = lambda index: steps[index]  # the data loader's workers are forked with it
    images = np.empty((len(steps), PICTURES, 224, 224, 3), np.uint8)
    actions = np.empty((len(steps), CHUNK, DIMS), np.float32)
    done = 0
    for samples in DataLoader(mixture, batch_size=batch, sampler=range(len(steps)), collate_fn=collate_fn,
                              num_workers=workers):
        pictures, chunks = batch_arrays(samples)
        images[done:done + len(samples)], actions[done:done + len(samples)] = pictures, chunks
        done += len(samples)
    return images, actions, episodes


def train(args) -> None:
    from omegaconf import OmegaConf

    from starVLA.dataloader.visual_prompt_datasets import get_vla_dataset_with_inline_vp

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists: use another --out")
    if not torch.cuda.is_available():
        raise SystemExit("train needs a GPU")
    if args.steps <= 0 or args.log_every <= 0:
        raise SystemExit("--steps and --log-every must be positive")
    cfg = OmegaConf.load(args.config)
    data_cfg = cfg.datasets.vla_data
    data_cfg.data_root_dir, data_cfg.visual_prompt_dir = args.data_root, args.visual_prompts
    data_cfg.data_mix = args.mix
    mixture = get_vla_dataset_with_inline_vp(data_cfg=data_cfg, seed=args.seed)
    (out / "checkpoints").mkdir(parents=True)
    mixture.save_dataset_statistics(out / "dataset_statistics.json")
    settings = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "func"}
    settings.update(tasks=list(TASKS), model="ResNet-18 (ImageNet weights, shared by both pictures) and an MLP",
                    loss="L1 on the 16 x 7 scaled action numbers")
    OmegaConf.save(OmegaConf.create(settings), out / "config.yaml")
    models, summary = {}, {}
    with open(out / "train_log.jsonl", "w") as log:
        for task in TASKS:
            t0 = time.time()
            images, actions, episodes = task_samples(mixture, task, args.batch, args.workers)
            print(f"{task}: {episodes} episodes, {len(images)} samples, made in {time.time() - t0:.0f} s", flush=True)
            images, actions = torch.from_numpy(images).cuda(), torch.from_numpy(actions).cuda()
            torch.manual_seed(args.seed)
            generator = torch.Generator(device="cuda").manual_seed(args.seed)
            model = SmallPolicy(pretrained=True, hidden=args.hidden).cuda().train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, args.steps, args.warmup))
            losses, t0 = [], time.time()
            for step in range(args.steps):
                rows = torch.randint(len(images), (args.batch,), generator=generator, device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predicted = model(random_shift(images[rows], args.shift, generator))
                loss = (predicted.float() - actions[rows]).abs().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                schedule.step()
                losses.append(loss.item())
                if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
                    row = {"task": task, "step": step + 1, "loss": float(np.mean(losses[-args.log_every:])),
                           "lr": schedule.get_last_lr()[0], "seconds": round(time.time() - t0, 1)}
                    print(json.dumps(row), flush=True)
                    log.write(json.dumps(row) + "\n")
                    log.flush()
            models[task] = model
            summary[task] = {"episodes": episodes, "samples": len(images), "last_loss": row["loss"],
                             "seconds": row["seconds"]}
            del images, actions
            torch.cuda.empty_cache()
    save(out / "checkpoints" / f"steps_{args.steps}_pytorch_model.pt", models, settings)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"saved {out / 'checkpoints' / f'steps_{args.steps}_pytorch_model.pt'}")


def serve(args) -> None:
    import logging

    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

    logging.basicConfig(level=logging.INFO, force=True)
    policy = load(args.ckpt, device="cuda" if torch.cuda.is_available() else "cpu")
    logging.info("small policies of %s on port %d", ", ".join(policy.models), args.port)
    WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=args.port, idle_timeout=args.idle_timeout,
                          metadata={"env": "simpler_env"}).serve_forever()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(required=True)
    p = commands.add_parser("train", help="train the 4 tasks' policies on VLA-P's demonstrations")
    p.add_argument("--config", required=True, help="the released run's config.yaml (its data settings)")
    p.add_argument("--data-root", required=True, help="the training sets' folder (select_demos.py --data-root)")
    p.add_argument("--visual-prompts", required=True, help="their visual prompts' folder")
    p.add_argument("--mix", default="vla_p", help="the mixture whose sets have the demonstrations (default: VLA-P's)")
    p.add_argument("--out", required=True, help="the run folder (RUN_ROOT/small); must not exist")
    p.add_argument("--steps", type=int, default=20000, help="training steps per task")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500, help="warm-up steps, then a cosine decay to 0")
    p.add_argument("--shift", type=int, default=8, help="the random shifts' largest move in pixels (0: none)")
    p.add_argument("--hidden", type=int, default=512, help="the MLP's width")
    p.add_argument("--workers", type=int, default=8, help="data loader workers for making the samples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=500)
    p.set_defaults(func=train)
    p = commands.add_parser("serve", help="serve the policies to VP-VLA's eval client")
    p.add_argument("--ckpt", required=True, help="RUN_ROOT/small/checkpoints/steps_N_pytorch_model.pt")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--idle-timeout", type=int, default=1800, help="seconds without requests before the server stops")
    p.set_defaults(func=serve)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
