"""The energy study (README, "Energy study"): the GPU energy per episode of the ASM system, the VLA alone and the
planner alone, from the energy runs.

    python -m examples.SimplerEnv.eval_files.planner.experiments.score_energy [--json FILE]

reads the study's energy runs (ENERGY=1; their names are below, or give --vla, --planner and --asm). Each has, for
every episode, the GPU's energy from the episode's start to its last step (`gpu_joules` in episode_costs.jsonl), the
GPU's idle power with the eval's servers loaded (idle_power.json), and next to it the same episodes with the null
controller (RUN_null): the simulation alone, with the same servers loaded. An episode's controller energy is its
energy minus its steps times the simulation's energy per step on its task (the null episodes' energy over their
steps): the GPU energy that the controller (its models, and the idle GPU while it plans) adds to the simulation.

The planned test: on harsh depth, the closed-loop ASM system against the VLA alone, controller energy per episode:
the exact two-sided paired sign-flip test at 0.05 (all sign patterns), with one value per episode index (the sum over
the 4 tasks of the difference). Only the episode indices with all 4 tasks in both evals count. The ASM system's
energy also comes offline (each episode's choice in the closed-loop eval, with the planner-only and VLA-only evals'
energies) as a check. Intervals: 95% percentile bootstrap over episode indices (10,000 draws).
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import runs, stats
from .score_costs import cost_records

# The study's energy runs (README, "Energy study"): the first 16 episodes of each task of iid15, one sim at a time.
VLA_RUN = "energy_vla_iid15"
PLANNER_RUNS = {"harsh": "energy_pseed31_iid15_harsh", "none": "energy_pseed31_iid15_none"}
ASM_RUN = "energy_closed_loop_iid15"  # one folder per depth level inside
PLANNED_LEVEL = "harsh"
# The server's CPU (AMD Threadripper PRO 7975WX) has a 350 W TDP over 32 cores; its energy counter (RAPL) can only be
# read as root, so the CPU energy is an estimate: the controller's CPU seconds times this.
CPU_WATTS_PER_CORE = 350.0 / 32
JOULES_PER_KWH = 3.6e6


@dataclass
class Episode:
    task: str
    index: int
    success: bool
    steps: int
    seconds: float
    joules: float  # the GPU's energy over the episode
    sim_joules: float  # the simulation's share: steps x the null episodes' energy per step on this task
    controller_seconds: float
    cpu_seconds: float
    idle_watts: float
    first_look_joules: Optional[float] = None  # planner records: the GPU's energy up to the end of the first look
    choice: Optional[str] = None  # closed-loop records: "planner" or "vla"

    @property
    def controller_joules(self) -> float:
        return self.joules - self.sim_joules

    @property
    def above_idle_joules(self) -> float:
        return self.controller_joules - self.idle_watts * self.controller_seconds

    @property
    def cpu_joules(self) -> float:
        return self.cpu_seconds * CPU_WATTS_PER_CORE


def sim_joules_per_step(null_root: str) -> Dict[str, float]:
    """task -> the null episodes' GPU energy per step (the simulation alone)."""
    joules: Dict[str, float] = {}
    steps: Dict[str, int] = {}
    for r in cost_records(null_root):
        if r.get("gpu_joules") is None:
            raise SystemExit(f"{null_root}: a null episode without gpu_joules (not an energy run?)")
        joules[r["env_name"]] = joules.get(r["env_name"], 0.0) + r["gpu_joules"]
        steps[r["env_name"]] = steps.get(r["env_name"], 0) + r["steps"]
    if not joules:
        raise SystemExit(f"{null_root}: no null episodes")
    return {t: joules[t] / steps[t] for t in joules}


def planner_records(root: str) -> Dict[Tuple[str, int], dict]:
    """(task, episode) -> its planner record (planner_stats.jsonl), for the first look and the ASM's choice."""
    out = {}
    for dirpath, _, files in os.walk(root):
        if "planner_stats.jsonl" in files:
            with open(os.path.join(dirpath, "planner_stats.jsonl")) as fh:
                for line in fh:
                    r = json.loads(line)
                    out[(r["env_name"], int(r["obj_episode_id"]))] = r
    return out


def load_episodes(root: str, null_root: str, idle_file: str) -> Dict[Tuple[str, int], Episode]:
    """The episodes of an energy run's folder (one depth level of a closed-loop eval), keyed by (task, episode)."""
    per_step = sim_joules_per_step(null_root)
    with open(idle_file) as fh:
        idle = json.load(fh)["watts"]
    plans = planner_records(root)
    out = {}
    for r in cost_records(root):
        task, index = r["env_name"], int(r["obj_episode_id"])
        if r.get("gpu_joules") is None:
            raise SystemExit(f"{root}: {task} episode {index} has no gpu_joules (not an energy run?)")
        if task not in per_step:
            raise SystemExit(f"{null_root}: no null episodes of {task}")
        plan = plans.get((task, index), {})
        out[(task, index)] = Episode(
            task=task, index=index, success=bool(r["success"]), steps=int(r["steps"]), seconds=float(r["seconds"]),
            joules=float(r["gpu_joules"]), sim_joules=int(r["steps"]) * per_step[task],
            controller_seconds=float(r["controller"]["seconds"]), cpu_seconds=float(r["controller"]["cpu_seconds"]),
            idle_watts=float(idle), first_look_joules=plan.get("first_look_gpu_joules"),
            choice=plan.get("selector_choice"))
    return out


def load_run(name: str, level: Optional[str] = None) -> Dict[Tuple[str, int], Episode]:
    """An energy run by name: its null episodes are in NAME_null and its idle power in NAME/idle_power.json; a
    closed-loop eval's episodes of one depth level are in NAME/LEVEL."""
    root = runs.path(name)
    return load_episodes(os.path.join(root, level) if level else root, root + "_null",
                         os.path.join(root, "idle_power.json"))


def offline_asm(asm: Dict[Tuple[str, int], Episode], planner: Dict[Tuple[str, int], Episode],
                vla: Dict[Tuple[str, int], Episode]) -> Dict[Tuple[str, int], Tuple[float, bool]]:
    """(controller energy, success) of the ASM system offline: each episode's choice in the closed-loop eval, the
    planner-only episode when it is the planner, else the planner-only first look plus the VLA-only episode."""
    out = {}
    for key, e in asm.items():
        if key not in planner or key not in vla:
            continue
        p = planner[key]
        if e.choice == "planner":
            out[key] = (p.controller_joules, p.success)
        elif e.choice == "vla" and p.first_look_joules is not None:
            out[key] = (p.first_look_joules + vla[key].controller_joules, vla[key].success)
    return out


def common_indices(*systems: Dict[Tuple[str, int], object]) -> List[int]:
    """The episode indices with every task of the first system in every system."""
    tasks = sorted({t for t, _ in systems[0]})
    indices = sorted({i for _, i in systems[0]})
    return [i for i in indices if all((t, i) in s for s in systems for t in tasks)]


def keys_of(system: Dict[Tuple[str, int], object], indices: List[int]) -> List[Tuple[str, int]]:
    return sorted(k for k in system if k[1] in set(indices))


def sign_flip_exact(v: np.ndarray, max_units: int = 20) -> float:
    """Exact two-sided paired sign-flip test of mean(v) = 0 over all 2^n sign patterns of the n values. With more than
    `max_units` values, the test of stats.sign_flip on the values in whole units."""
    v = np.asarray(v, float)
    if len(v) > max_units:
        return stats.sign_flip(np.round(v), 1)
    signs = ((np.arange(2 ** len(v))[:, None] >> np.arange(len(v))) & 1) * 2 - 1
    sums = np.abs(signs @ np.abs(v))
    return float(np.mean(sums >= abs(v.sum()) * (1 - 1e-12)))


def paired_test(new: Dict[Tuple[str, int], float], base: Dict[Tuple[str, int], float],
                indices: List[int]) -> dict:
    """new - base per episode: the exact sign-flip test on the per-index sums, the mean per episode and its 95%
    bootstrap interval over episode indices."""
    keys = [k for k in sorted(new) if k[1] in set(indices)]
    d = np.array([new[k] - base[k] for k in keys])
    units = np.array([k[1] for k in keys])
    sums = stats.unit_sums(d, units)
    lo, hi = stats.cluster_bootstrap_interval(d, units)
    return {"mean": float(d.mean()), "interval": (lo, hi), "p": sign_flip_exact(sums), "indices": len(indices),
            "episodes": len(keys), "lower": int((sums < 0).sum()), "higher": int((sums > 0).sum())}


def ratio_interval(num: np.ndarray, den: np.ndarray, units: np.ndarray, draws: int = 10000,
                   seed: int = 0) -> Tuple[float, float]:
    """95% bootstrap interval of sum(num) / sum(den) over episode indices (inf when a draw has den 0)."""
    names = sorted(set(units.tolist()))
    n_sums, d_sums = stats.unit_sums(num, units), stats.unit_sums(den, units)
    idx = np.random.default_rng(seed).integers(0, len(names), size=(draws, len(names)))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = n_sums[idx].sum(1) / d_sums[idx].sum(1)
    # order statistics, not np.quantile: its interpolation between two infinite draws gives nan
    r = np.sort(np.where(np.isfinite(r), r, np.inf))
    return float(r[int(0.025 * (draws - 1))]), float(r[int(np.ceil(0.975 * (draws - 1)))])


def summary(energy: Dict[Tuple[str, int], float], success: Dict[Tuple[str, int], bool],
            indices: List[int], extra: Optional[Dict[str, Dict[Tuple[str, int], float]]] = None) -> dict:
    """A system's means over the episodes of `indices`: controller energy (J, with interval), success, energy per
    success, kWh per million episodes, and the means of `extra` (name -> per-episode values)."""
    keys = [k for k in sorted(energy) if k[1] in set(indices)]
    e = np.array([energy[k] for k in keys])
    y = np.array([float(success[k]) for k in keys])
    units = np.array([k[1] for k in keys])
    out = {"episodes": len(keys), "joules": float(e.mean()), "interval": stats.cluster_bootstrap_interval(e, units),
           "success": float(y.mean()), "kwh_per_million": float(e.mean()) * 1e6 / JOULES_PER_KWH,
           "joules_per_success": float(e.sum() / y.sum()) if y.sum() else float("inf"),
           "joules_per_success_interval": ratio_interval(e, y, units)}
    for name, values in (extra or {}).items():
        out[name] = float(np.mean([values[k] for k in keys]))
    return out


def per_episode(system: Dict[Tuple[str, int], Episode]) -> Dict[str, Dict[Tuple[str, int], float]]:
    return {"total_joules": {k: e.joules for k, e in system.items()},
            "above_idle_joules": {k: e.above_idle_joules for k, e in system.items()},
            "cpu_joules": {k: e.cpu_joules for k, e in system.items()},
            "seconds": {k: e.seconds for k, e in system.items()}}


def run_facts(system: Dict[Tuple[str, int], Episode]) -> dict:
    """An energy run's idle power (W) and the simulation's energy per step on each task (J)."""
    per_step = {}
    for e in system.values():
        per_step.setdefault(e.task, e.sim_joules / e.steps if e.steps else 0.0)
    return {"idle_watts": next(iter(system.values())).idle_watts, "sim_joules_per_step": per_step}


def score(vla: Dict[Tuple[str, int], Episode], planner: Dict[str, Dict[Tuple[str, int], Episode]],
          asm: Dict[str, Dict[Tuple[str, int], Episode]]) -> dict:
    """Every number of the study: the planned test (on PLANNED_LEVEL) and, per depth level, each system's summary."""
    out = {"levels": {}}
    for level in sorted(asm, key=lambda lv: lv != PLANNED_LEVEL):
        a, p = asm[level], planner.get(level, {})
        idx = common_indices(a, vla, p) if p else common_indices(a, vla)
        energy = {"VLA alone": {k: e.controller_joules for k, e in vla.items()},
                  "ASM system": {k: e.controller_joules for k, e in a.items()}}
        success = {"VLA alone": {k: e.success for k, e in vla.items()},
                   "ASM system": {k: e.success for k, e in a.items()}}
        extra = {"VLA alone": per_episode(vla), "ASM system": per_episode(a)}
        if p:
            energy["Planner alone"] = {k: e.controller_joules for k, e in p.items()}
            success["Planner alone"] = {k: e.success for k, e in p.items()}
            extra["Planner alone"] = per_episode(p)
            off = offline_asm(a, p, vla)
            # The oracle after the fact: the planner when it succeeds or when the VLA fails too, else the VLA.
            oracle = {k: (p[k].controller_joules, p[k].success) if p[k].success or not vla[k].success
                      else (p[k].first_look_joules + vla[k].controller_joules, True)
                      for k in p if k in vla and p[k].first_look_joules is not None}
            for name, values in (("ASM system (offline)", off), ("Oracle", oracle)):
                if all((t, i) in values for t, i in keys_of(a, idx)):
                    energy[name] = {k: v[0] for k, v in values.items()}
                    success[name] = {k: v[1] for k, v in values.items()}
        level_out = {"indices": idx, "systems": {n: summary(energy[n], success[n], idx, extra.get(n)) for n in energy},
                     "vla_picks": sum(a[k].choice == "vla" for k in keys_of(a, idx)),
                     "runs": {"VLA alone": run_facts(vla), "ASM system": run_facts(a),
                              **({"Planner alone": run_facts(p)} if p else {})}}
        for s in level_out["systems"].values():
            s["saving_vs_vla"] = 1 - s["joules"] / level_out["systems"]["VLA alone"]["joules"]
        level_out["test"] = paired_test(energy["ASM system"], energy["VLA alone"], idx)
        out["levels"][level] = level_out
    if PLANNED_LEVEL in out["levels"]:
        t = out["levels"][PLANNED_LEVEL]["test"]
        out["planned"] = {**t, "win": t["p"] < 0.05 and t["mean"] < 0}
    return out


def print_report(result: dict) -> None:
    if "planned" in result:
        t = result["planned"]
        print(f"Planned test (harsh depth, controller GPU energy per episode, ASM system - VLA alone, "
              f"{t['indices']} episode indices, {t['episodes']} episodes): {t['mean'] / 1000:+.2f} kJ "
              f"(95% interval {t['interval'][0] / 1000:+.2f} to {t['interval'][1] / 1000:+.2f}), "
              f"{t['lower']} indices lower / {t['higher']} higher, exact p = {t['p']:.2g}: "
              + ("a win (the ASM system uses less)" if t["win"] else "not a win"))
    for level, lv in result["levels"].items():
        print(f"\n## {runs.LEVEL_NAMES.get(level, level)} depth ({len(lv['indices'])} episode indices; the ASM gave "
              f"{lv['vla_picks']} episodes to the VLA)\n")
        print("| System | Episodes | Success | Controller GPU energy (kJ) | Saving vs VLA | Per success (kJ) | "
              "kWh per million episodes | Total GPU (kJ) | Above idle (kJ) | CPU estimate (J) | Seconds |")
        print("|---" * 11 + "|")
        for name, s in lv["systems"].items():
            lo, hi = s["interval"]
            plo, phi = s["joules_per_success_interval"]
            more = " | ".join("n/a" if s.get(k) is None else fmt for k, fmt in (
                ("total_joules", f"{s.get('total_joules', 0) / 1000:.2f}"),
                ("above_idle_joules", f"{s.get('above_idle_joules', 0) / 1000:.2f}"),
                ("cpu_joules", f"{s.get('cpu_joules', 0):.0f}"), ("seconds", f"{s.get('seconds', 0):.1f}")))
            print(f"| {name} | {s['episodes']} | {100 * s['success']:.1f}% | {s['joules'] / 1000:.2f} "
                  f"({lo / 1000:.2f} to {hi / 1000:.2f}) | {100 * s['saving_vs_vla']:.1f}% | "
                  f"{s['joules_per_success'] / 1000:.2f} ({plo / 1000:.2f} to {phi / 1000:.2f}) | "
                  f"{s['kwh_per_million']:,.0f} | {more} |")
        print("\n| Energy run | Idle power (W) | Simulation energy per step (J) |\n|---|---|---|")
        for name, f in lv["runs"].items():
            per_step = ", ".join(f"{runs.SHORT.get(t, t)} {j:.1f}" for t, j in sorted(f["sim_joules_per_step"].items()))
            print(f"| {name} | {f['idle_watts']:.1f} | {per_step} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vla", default=VLA_RUN)
    ap.add_argument("--planner", nargs="*", default=[f"{k}={v}" for k, v in PLANNER_RUNS.items()],
                    help="LEVEL=RUN for each depth level")
    ap.add_argument("--asm", default=ASM_RUN, help="the closed-loop energy run (one folder per depth level)")
    ap.add_argument("--levels", nargs="+", default=["harsh", "none"])
    ap.add_argument("--json", default=None, help="also write every number here")
    a = ap.parse_args()
    planner = {}
    for item in a.planner:
        level, name = item.split("=", 1)
        planner[level] = load_run(name)
    vla = load_run(a.vla)
    asm = {level: load_run(a.asm, level) for level in a.levels}
    result = score(vla, planner, asm)
    print_report(result)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(result, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
