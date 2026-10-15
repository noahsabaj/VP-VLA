"""A closed-loop ASM eval (selector_eval.sh) against the offline eval, episode by episode.

The offline result of an episode: the ASM's pick from the planner-only eval's first-look features (same layout,
same depth-noise draw), and the outcome of the picked controller there (planner: the planner-only eval; VLA: the
VLA-only eval, whose server settings the closed-loop eval uses). Videos are compared byte for byte.

    python -m examples.SimplerEnv.eval_files.planner.experiments.check_closed_loop live_pseed \\
        --planner 'pseed_paper_{}' --models final_pseed [--vla fast3_train_g0 fast3_train_g1]

--planner: the planner-only evals, a run name with {} for the depth level; --models: the ASM files the closed-loop
eval used, a file name with {} for the depth level (e.g. two_model/asm_{}.json), or a folder with the
sel_v3_{level}.json models.
"""

import argparse
import glob
import hashlib
import json
import os

import numpy as np

from .. import selector as s
from . import runs
from .runs import LEVEL_NAMES, LEVELS


def videos(roots):
    """(task, episode) -> (success, md5) for every video under the roots; two videos of one try stop the check."""
    out = {}
    for root in roots:
        for task in s.TASKS:
            for f in sorted(glob.glob(os.path.join(root, task, "**", "*.mp4"), recursive=True)):
                m = s.VIDEO.search(f.replace("\\", "/"))
                if not m:
                    continue
                key = (task, int(m.group(2)))
                if key in out:
                    raise SystemExit(f"two videos for {key} under {', '.join(roots)}")
                with open(f, "rb") as fh:
                    out[key] = (m.group(1) == "success", hashlib.md5(fh.read()).hexdigest())
    return out


def live_records(root):
    out = {}
    for task in s.TASKS:
        for path in sorted(glob.glob(os.path.join(root, task, "**", "planner_stats.jsonl"), recursive=True)):
            with open(path) as fh:
                lines = list(fh)
            for n, line in enumerate(lines, 1):
                r = json.loads(line)
                if r.get("mode") != "selector":
                    raise SystemExit(f"{path}:{n}: mode {r.get('mode')!r}, not a live ASM try")
                key = (task, r["obj_episode_id"])
                if key in out:
                    raise SystemExit(f"{path}:{n}: a second record for {key}")
                out[key] = r
    return out


def probs(decision, prefix=""):
    """An ASM decision's probabilities: "P 0.1234" (P(planner)), and P(VLA) for the two-model ASM."""
    text = f"P {decision[prefix + 'p_planner']:.4f}"
    return text + (f", P(VLA) {decision[prefix + 'p_vla']:.4f}" if prefix + "p_vla" in decision else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="the closed-loop eval (selector_eval.sh's NAME)")
    ap.add_argument("--planner", required=True, help="planner-only evals: a run name with {} for the depth level")
    ap.add_argument("--models", required=True,
                    help="the ASM files it used: a file name with {} for the depth level, or a folder with the "
                         "sel_v3_{level}.json models")
    ap.add_argument("--vla", nargs="+", default=runs.FAST_VLA["paper"], help="VLA-only evals of the same layouts")
    a = ap.parse_args()
    live_root, planner_runs, models = runs.path(a.run), runs.path(a.planner), runs.path(a.models)
    model_files = models if "{}" in models else f"{models}/sel_v3_{{}}.json"
    vla_runs = runs.paths(a.vla)
    vla_vid = videos(vla_runs)
    print(f"Live run {live_root}; offline: {planner_runs} and {' '.join(vla_runs)}; models {models}")
    print("| Depth | Tries | Same first-look features | Same pick | Max P difference | VLA picks live / offline | "
          "Live success | Offline success | Same outcome | VLA tries: same video | Planner tries: same video |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    notes = []
    for lvl in [lvl for lvl in LEVELS if os.path.isdir(os.path.join(live_root, lvl))]:
        asm = s.load_asm(model_files.format(lvl))
        live, live_vid = live_records(f"{live_root}/{lvl}"), videos([f"{live_root}/{lvl}"])
        plan_root = planner_runs.format(lvl)
        plan, plan_vid = s.load_planner(plan_root), videos([plan_root])
        keys = sorted(k for k in live if k in plan and k in vla_vid)
        if len(keys) < len(live):
            notes.append(f"{LEVEL_NAMES[lvl]}: {len(live) - len(keys)} live tries have no offline result")
        off = {k: asm.decide(plan[k], k[0]) for k in keys}
        pick_off = {k: off[k]["choice"] for k in keys}
        out_off = {k: bool(plan[k]["success"]) if pick_off[k] == "planner" else vla_vid[k][0] for k in keys}
        out_live = {k: bool(live[k]["success"]) for k in keys}
        same_feat = sum(np.array_equal(s.row_features(live[k], runs.CHECK), s.row_features(plan[k], runs.CHECK))
                        for k in keys)
        same_pick = sum(live[k]["selector_choice"] == pick_off[k] for k in keys)
        max_dp = max(abs(live[k][f"selector_{n}"] - off[k][n]) for k in keys for n in off[k] if n != "choice")
        vla_keys = [k for k in keys if live[k]["selector_choice"] == "vla"]
        plan_keys = [k for k in keys if live[k]["selector_choice"] == "planner"]
        same_vla = sum(live_vid.get(k, (None, None))[1] == vla_vid[k][1] for k in vla_keys)
        same_plan = sum(live_vid.get(k, (None, None))[1] == plan_vid.get(k, (None, ""))[1] for k in plan_keys)
        bad_video = [k for k in keys if k in live_vid and live_vid[k][0] != out_live[k]]
        print(f"| {LEVEL_NAMES[lvl]} | {len(keys)} | {same_feat} | {same_pick} | {max_dp:.2g} | {len(vla_keys)} / "
              f"{sum(v == 'vla' for v in pick_off.values())} | {100 * np.mean(list(out_live.values())):.1f} | "
              f"{100 * np.mean(list(out_off.values())):.1f} | {sum(out_live[k] == out_off[k] for k in keys)} | "
              f"{same_vla} of {len(vla_keys)} | {same_plan} of {len(plan_keys)} |")
        for k in keys:
            if live[k]["selector_choice"] != pick_off[k] or out_live[k] != out_off[k]:
                notes.append(f"{LEVEL_NAMES[lvl]} {k[0].split('-')[0]} episode {k[1]}: live pick "
                             f"{live[k]['selector_choice']} ({probs(live[k], 'selector_')}) -> "
                             f"{'success' if out_live[k] else 'failure'}; offline pick {pick_off[k]} "
                             f"({probs(off[k])}) -> {'success' if out_off[k] else 'failure'}")
        if bad_video:
            notes.append(f"{LEVEL_NAMES[lvl]}: video name and record disagree on {bad_video}")
    print("\n".join(notes) if notes else "Every live try agrees with its offline result.")


if __name__ == "__main__":
    main()
