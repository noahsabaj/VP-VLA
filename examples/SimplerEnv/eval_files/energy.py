"""GPU energy for the energy runs (experiments README, "Energy study").

The GPU's energy counter (NVML's total energy consumption: the joules the GPU used since its driver loaded) is read
through the driver's own library (libnvidia-ml), so no Python package is needed. One read takes about 6 ms. Over a
window of a few seconds it agrees with the GPU's power readings, but it advances in steps of about 0.1 s, so the
energy runs read it per episode (model_time.py: when the episode starts and after its last step), not per request.
A window's energy is the whole GPU's: it is the episode's own only with one sim at a time on a GPU that runs nothing
else, which is how the energy runs run.

NullController does nothing (no motion, gripper open, never stops), so an episode of it is the simulation alone:
the energy runs subtract its energy per step from each episode, with the same servers loaded.

    python -m examples.SimplerEnv.eval_files.energy idle --gpu 0 --seconds 60

prints the GPU's mean power over SECONDS (after --settle seconds), as JSON: the runners' idle_power.json.
"""

import argparse
import ctypes
import json
import os
import time
from typing import Optional

import numpy as np

GPU_ENV = "ENERGY_GPU"  # the GPU whose counter the sims read (the runners set it with ENERGY=1)


class GPUEnergy:
    """One GPU's energy counter (NVML). `lib` replaces the driver's library (tests)."""

    def __init__(self, index: int, lib=None) -> None:
        self.index = index
        self._lib = lib if lib is not None else ctypes.CDLL("libnvidia-ml.so.1")
        _check(self._lib.nvmlInit_v2(), "nvmlInit_v2")
        self._handle = ctypes.c_void_p()
        _check(self._lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(self._handle)),
               "nvmlDeviceGetHandleByIndex_v2")

    def joules(self) -> float:
        """The joules the GPU used since its driver loaded."""
        value = ctypes.c_ulonglong()
        _check(self._lib.nvmlDeviceGetTotalEnergyConsumption(self._handle, ctypes.byref(value)),
               "nvmlDeviceGetTotalEnergyConsumption")
        return value.value / 1000.0  # NVML counts millijoules


def _check(code: int, name: str) -> None:
    if code != 0:
        raise RuntimeError(f"NVML {name} failed with code {code}")


def energy_from_env() -> Optional[GPUEnergy]:
    """The counter of the GPU named in ENERGY_GPU, or None when it is not set (every run but the energy runs)."""
    index = os.environ.get(GPU_ENV, "").strip()
    return GPUEnergy(int(index)) if index else None


def idle_power(counter: GPUEnergy, seconds: float, settle: float = 0.0, clock=time.perf_counter,
               sleep=time.sleep) -> dict:
    """The GPU's mean power over `seconds`, after `settle` seconds of waiting (so that it reaches its idle state)."""
    sleep(settle)
    j0, t0 = counter.joules(), clock()
    sleep(seconds)
    joules, elapsed = counter.joules() - j0, clock() - t0
    return {"gpu": counter.index, "seconds": elapsed, "joules": joules, "watts": joules / elapsed}


class NullController:
    """A controller that does nothing: no motion, gripper open, never stops (the evaluator's model interface)."""

    def reset(self, task_description: str) -> None:
        pass

    def step(self, image, task_description: Optional[str] = None, **kwargs):
        raw_action = {"world_vector": np.zeros(3), "rotation_delta": np.zeros(3), "open_gripper": np.ones(1)}
        action = {"world_vector": np.zeros(3), "rot_axangle": np.zeros(3), "gripper": np.ones(1),
                  "terminate_episode": np.zeros(1)}
        return raw_action, action

    def visualize_epoch(self, predicted_raw_actions, images, save_path: str) -> None:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    idle = sub.add_parser("idle", help="the GPU's mean power over a window, as JSON")
    idle.add_argument("--gpu", type=int, required=True)
    idle.add_argument("--seconds", type=float, default=60.0)
    idle.add_argument("--settle", type=float, default=30.0, help="seconds to wait first")
    a = ap.parse_args()
    print(json.dumps(idle_power(GPUEnergy(a.gpu), a.seconds, a.settle)))


if __name__ == "__main__":
    main()
