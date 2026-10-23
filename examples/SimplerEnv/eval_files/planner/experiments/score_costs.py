"""What each eval cost: per episode, its wall-clock time, its model requests (model_time.py) and its GPU energy.

    python -m examples.SimplerEnv.eval_files.planner.experiments.score_costs RUN [RUN...]

reads each run folder's cost records (`episode_costs.jsonl`, one per sim process, written by planner_evaluator.py
since 2026-10-23) and the GPU monitor of its runner (`dmon.txt`: nvidia-smi dmon with power and timestamps, written
by planner_eval.sh, vla_eval.sh and selector_eval.sh since 2026-10-23). The GPU energy per episode is the energy of
the whole eval (power x time, from the start of the episodes to the end) over its episodes. It is the eval's cost
only when the eval had the GPU to itself: one eval at a time (planner_evals.sh with AT_ONCE=1) and no other user.
The request times are round trips (model_time.py): GPU time with one sim at a time, waiting time when sims share
the servers.
"""

import argparse
import glob
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import runs

KINDS = ["vla", "vlm", "sam"]


def read_dmon(path: str) -> List[Tuple[datetime, float]]:
    """(time, GPU power in W) for each sample of an `nvidia-smi dmon -s pum -o DT` log; empty when the log has no
    power or no timestamps (the runners before 2026-10-23 logged neither)."""
    out, cols = [], None
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                names = line[1:].split()
                if "pwr" in names and "Date" in names and "Time" in names:
                    cols = names  # the units line that follows has no "pwr"
                continue
            v = line.split()
            if cols is None or len(v) != len(cols):
                continue
            try:
                t = datetime.strptime(f"{v[cols.index('Date')]} {v[cols.index('Time')]}", "%Y%m%d %H:%M:%S")
                out.append((t, float(v[cols.index("pwr")])))
            except ValueError:  # "-" where the GPU gave no value
                continue
    return out


def gpu_energy(path: str) -> Optional[Tuple[float, float]]:
    """(joules, seconds) of a dmon log: each sample's power for the time until the next sample (the last one for the
    usual interval). None without at least 2 samples with power."""
    samples = read_dmon(path) if os.path.exists(path) else []
    if len(samples) < 2:
        return None
    t = np.array([(s[0] - samples[0][0]).total_seconds() for s in samples])
    p = np.array([s[1] for s in samples])
    dt = np.diff(t)
    dt = np.append(dt, np.median(dt))
    return float((p * dt).sum()), float(dt.sum())


def cost_records(root: str) -> List[dict]:
    """Every cost record under a run folder, one per episode: a second record for the same task and episode stops
    the load (a rerun written into the same folder)."""
    records, seen = [], set()
    for path in sorted(glob.glob(os.path.join(root, "**", "episode_costs.jsonl"), recursive=True)):
        with open(path) as fh:
            for n, line in enumerate(fh, 1):
                r = json.loads(line)
                key = (r["env_name"], r["obj_episode_id"], os.path.relpath(path, root).split(os.sep)[0])
                if key in seen:
                    raise SystemExit(f"{path}:{n}: a second record for {r['env_name']} episode {r['obj_episode_id']}")
                seen.add(key)
                records.append(r)
    return records


def first_looks(root: str) -> List[dict]:
    """The planner records' first-look costs (planner_client.py), for the planner-only and closed-loop evals."""
    out = []
    for path in sorted(glob.glob(os.path.join(root, "**", "planner_stats.jsonl"), recursive=True)):
        with open(path) as fh:
            out += [r for r in map(json.loads, fh) if "first_look_model" in r]
    return out


def model_mean(records: List[dict], key: str = "model") -> Dict[str, Tuple[float, float]]:
    """kind -> (mean requests, mean round-trip seconds) per record; a record with none of a kind counts as 0."""
    return {k: (float(np.mean([r[key].get(k, {}).get("calls", 0) for r in records])),
                float(np.mean([r[key].get(k, {}).get("seconds", 0.0) for r in records]))) for k in KINDS}


def run_cost(name: str) -> dict:
    """One eval's costs: episodes, success rate, seconds per episode, model requests per episode, GPU energy per
    episode (None without a dmon log with power) and the first look's costs (planner records only)."""
    root = runs.path(name)
    records = cost_records(root)
    if not records:
        raise SystemExit(f"{root}: no cost records (episode_costs.jsonl)")
    energy = gpu_energy(os.path.join(root, "dmon.txt"))
    looks = first_looks(root)
    return {"episodes": len(records), "success": float(np.mean([r["success"] for r in records])),
            "seconds": float(np.mean([r["seconds"] for r in records])), "model": model_mean(records),
            "energy_per_episode": None if energy is None else energy[0] / len(records),
            "gpu_seconds": None if energy is None else energy[1],
            "mean_watts": None if energy is None else energy[0] / energy[1],
            "first_look": None if not looks else {
                "episodes": len(looks), "seconds": float(np.mean([r["first_look_seconds"] for r in looks])),
                "model": model_mean(looks, "first_look_model")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folders (names in $RUNS_DIR, or paths)")
    a = ap.parse_args()
    costs = {name: run_cost(name) for name in a.runs}
    print("Per episode (model requests: count / round-trip seconds):\n")
    print("| Eval | Episodes | Success | Seconds | " + " | ".join(f"{k.upper()} requests" for k in KINDS)
          + " | GPU energy (kJ) | Mean GPU power (W) | Monitored (min) |")
    print("|---" * (7 + len(KINDS)) + "|")
    for name, c in costs.items():
        energy = "n/a | n/a | n/a" if c["energy_per_episode"] is None else \
            f"{c['energy_per_episode'] / 1000:.2f} | {c['mean_watts']:.0f} | {c['gpu_seconds'] / 60:.0f}"
        print(f"| {name} | {c['episodes']} | {100 * c['success']:.1f}% | {c['seconds']:.1f} | "
              + " | ".join(f"{n:.1f} / {sec:.2f}" for n, sec in (c["model"][k] for k in KINDS)) + f" | {energy} |")
    looks = {name: c["first_look"] for name, c in costs.items() if c["first_look"]}
    if looks:
        print("\nThe planner's first look per episode (what the ASM needs in every episode):\n")
        print("| Eval | Episodes | Seconds | " + " | ".join(f"{k.upper()} requests" for k in KINDS) + " |")
        print("|---" * (3 + len(KINDS)) + "|")
        for name, f in looks.items():
            print(f"| {name} | {f['episodes']} | {f['seconds']:.1f} | "
                  + " | ".join(f"{n:.1f} / {sec:.2f}" for n, sec in (f["model"][k] for k in KINDS)) + " |")


if __name__ == "__main__":
    main()
