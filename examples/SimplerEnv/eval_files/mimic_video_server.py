"""A policy server for mimic-video in our evaluator (the experiments README, "mimic-video").

mimic-video (github.com/mimic-video/mimic-video) is a video-action model: a video model (Cosmos-Predict2) predicts
how the scene goes on, and an action decoder reads the video model's hidden states and gives a chunk of actions.

    MIMIC_PYTHON -m examples.SimplerEnv.eval_files.mimic_video_server --checkpoints DIR --port PORT

Run it from this repository's root with the Python of mimic-video's own environment (its `model/.venv`, with its
SimplerEnv fork installed as in its README, and `websockets` and `msgpack` added). Our eval client
(mimic_video_client.py) runs in the simulator's environment and talks to it as to VP-VLA's policy server.

The policy is mimic-video's own inference class (`VAMInference` of its SimplerEnv fork), not a copy: each episode
gets an instance of it (a session) with its own pictures, states and action chunk, and all sessions share the one
loaded model. So an episode through this server gets the actions that mimic-video's own eval loop would give it.
The requests of several simulators are answered one at a time. The model's noise has a fixed seed in mimic-video's
code, so an episode does not depend on the other episodes or on the order of the requests.

The checkpoint's files, from DIR (mimic-video's `scripts/download_checkpoints.py`): by default its Bridge model with
the fine-tuned video backbone, as in its `eval/bridge/eval.sh` ("ftcosmos"). --experiment, --video-model,
--action-model and --stats name another one.
"""

import argparse
import copy
import logging
import os
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np

# mimic-video's Bridge model with the fine-tuned video backbone (its eval/bridge/eval.sh, "ftcosmos").
VIDEO_MODEL = "v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused"
EXPERIMENT = f"w2a_bridge_{VIDEO_MODEL}_lr1.000e-04_layer20_bsz256"
ACTION_ITERATION = "iter_000014112"
ACTION_KEYS = ("world_vector", "rot_axangle", "gripper", "terminate_episode")


class Sessions:
    """The episodes' policies, as the websocket server's policy (predict_action).

    make(): a new policy with mimic-video's interface: reset(task) and step(image, task, pose, gripper), where pose
    has the hand's position `p` and quaternion `q` (w first) in the robot's base frame, and step gives the
    simulator's action. At most `limit` sessions are kept: a simulator that stops without ending its session leaves
    one behind, and the oldest goes first.
    """

    def __init__(self, make: Callable[[], object], limit: int = 64) -> None:
        self._make = make
        self._limit = limit
        self._sessions: "OrderedDict[str, object]" = OrderedDict()

    def __len__(self) -> int:
        return len(self._sessions)

    def predict_action(self, op: str, session: str, **request) -> dict:
        if op == "reset":
            policy = self._make()
            policy.reset(request["task"])
            self._sessions.pop(session, None)
            self._sessions[session] = policy
            while len(self._sessions) > self._limit:
                self._sessions.popitem(last=False)
            return {}
        if op == "end":
            self._sessions.pop(session, None)
            return {}
        if op != "step":
            raise ValueError(f"unknown request: {op!r}")
        if session not in self._sessions:
            raise KeyError(f"no session {session!r}: reset comes first")
        policy = self._sessions[session]
        new_chunk = policy.action_buffer is None  # this step asks the model for the next action chunk
        pose = SimpleNamespace(p=request["ee_p"], q=request["ee_q"])
        t0 = time.perf_counter()
        action = policy.step(request["image"], request["task"], pose, request["gripper"])
        seconds = time.perf_counter() - t0
        if new_chunk:
            logging.info("session %s: a new action chunk in %.2f s", session[:8], seconds)
        return {**{k: np.asarray(action[k]) for k in ACTION_KEYS}, "new_chunk": new_chunk, "seconds": seconds}


def checkpoint_files(checkpoints: Path, experiment: str, video_model, action_model, stats) -> dict:
    """The model's files; the ones that are not given are the Bridge model's in `checkpoints`."""
    files = {
        "video_model": Path(video_model or checkpoints / "video_backbone" / f"{VIDEO_MODEL}.pt"),
        "action_model": Path(action_model or checkpoints / "action_decoder" / f"{experiment}_{ACTION_ITERATION}.pt"),
        "stats": Path(stats or checkpoints / "dataset_statistics" / "bridge.json"),
    }
    missing = [str(f) for f in files.values() if not f.is_file()]
    if missing:
        raise SystemExit("mimic_video_server: missing: " + ", ".join(missing))
    return files


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoints", type=Path, required=True, help="the folder of mimic-video's checkpoints")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1", help="the address it listens on (default: this machine only)")
    parser.add_argument("--experiment", default=EXPERIMENT, help="the model's experiment name in mimic-video's configs")
    parser.add_argument("--video-model", help="the video backbone's file (default: the Bridge model's)")
    parser.add_argument("--action-model", help="the action decoder's file (default: the Bridge model's)")
    parser.add_argument("--stats", help="the dataset statistics' file (default: the Bridge model's)")
    parser.add_argument("--img-horizon", type=int, default=5, help="pictures the model reads (eval.sh: 5)")
    parser.add_argument("--lowdim-horizon", type=int, default=1, help="states the model reads (eval.sh: 1)")
    parser.add_argument("--stop-step", type=int, default=0,
                        help="the video denoising step whose hidden states the action decoder reads (0: one pass of "
                             "the video model, the paper's default)")
    parser.add_argument("--execute", type=int, default=5, help="actions of each chunk that the robot does (eval.sh: 5)")
    parser.add_argument("--idle-timeout", type=int, default=1800, help="seconds without requests before it stops")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, force=True)
    files = checkpoint_files(args.checkpoints, args.experiment, args.video_model, args.action_model, args.stats)
    # mimic-video's configs read the checkpoints' folder from this variable when they are imported.
    os.environ.setdefault("COSMOS_PREDICT2_ARGS", f"--checkpoints {args.checkpoints}")

    from simpler_env.policies.vam.video_action_model import VAMInference  # mimic-video's SimplerEnv fork

    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

    t0 = time.perf_counter()
    loaded = VAMInference(args.experiment, str(files["video_model"]), str(files["action_model"]), files["stats"],
                          args.img_horizon, args.lowdim_horizon, args.stop_step, args.execute, is_hil=False)
    logging.info("mimic-video %s loaded in %.0f s; port %d", args.experiment, time.perf_counter() - t0, args.port)
    # A session is a shallow copy: it shares the model, and reset() gives it an episode's state of its own.
    sessions = Sessions(lambda: copy.copy(loaded))
    WebsocketPolicyServer(policy=sessions, host=args.host, port=args.port, idle_timeout=args.idle_timeout,
                          metadata={"env": "simpler_env", "policy": "mimic-video"}).serve_forever()


if __name__ == "__main__":
    main()
