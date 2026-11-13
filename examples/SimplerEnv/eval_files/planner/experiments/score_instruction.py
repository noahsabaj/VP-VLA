"""The instruction study (README.md, "Instruction study", has the plan that this scorer follows).

In one scene with the same layouts (the choice scenes, eval_files/choice_scenes.py), does a controller do what the
instruction says when the instruction pairs an object with a target that it was never paired with? Two of the four
tasks are pairings that the training sets have (the spoon on the towel, the carrot on the plate) and two are new (the
spoon on the plate, the carrot on the towel).

    python -m examples.SimplerEnv.eval_files.planner.experiments.score_instruction [--layouts iid19]

Evals (CONTROLLERS, in $RUNS_DIR): VLA-0 (vla_choice_{layouts}), VLA-PL (vlapl_choice_{layouts}), the planner with
the fixed rule (pseed31_choice_{layouts}_none) and the planner reading the VLM's marks (vp_choice_{layouts}_none).
Each episode's outcome and its stats after the last step (the goal and the logged pair) come from the record that the
evaluator writes for every controller (episode_costs.jsonl). Only the design's episodes count (EPISODES of each
task). The four tasks share the layout at an episode index, so every test takes one value per episode index, and only
the indices that every eval of the test has in full (all four tasks). A test without all its evals is not run, and
the evals it lacks are named.
"""

import argparse
import os
import re
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from examples.SimplerEnv.eval_files import choice_scenes as cs
from examples.SimplerEnv.eval_files import harder_scenes as hs

from . import runs
from .score_costs import cost_records
from .stats import bootstrap_interval, holm, sign_flip

EPISODES = 48  # the design's episodes of each task: 0 to 47
VLA0, VLAPL, RULE, MARKS = "VLA-0", "VLA-PL", "planner (fixed rule)", "planner (VLM marks)"
# Each controller's eval, {} for the layout set.
CONTROLLERS = {VLA0: "vla_choice_{}", VLAPL: "vlapl_choice_{}", RULE: "pseed31_choice_{}_none",
               MARKS: "vp_choice_{}_none"}
TASKS = [*cs.TRAINED, *cs.NEW]
ALPHA = 0.05

Arm = Dict[Tuple[str, int], dict]  # (task, episode) -> the episode's record


def load(name: str, layouts: str) -> Optional[Arm]:
    """An eval's records of the design's episodes; None when the eval is not there. A record of another layout set,
    or a second record for an episode, stops the load."""
    root = runs.path(name)
    if not os.path.isdir(root):
        return None
    seed = re.fullmatch(r"iid(\d+)", layouts)
    if not seed:
        raise SystemExit(f"the choice scenes have a random layout per episode only: --layouts iidN, not {layouts!r}")
    out = {}
    for r in cost_records(root):
        key = (r["env_name"], int(r["obj_episode_id"]))
        if key[0] not in cs.TASKS:
            continue
        if r.get("layout_seed") != int(seed.group(1)) or not r.get("layout_per_episode"):
            raise SystemExit(f"{name}: {key[0]} episode {key[1]} is not of the layout set {layouts} (layout seed "
                             f"{r.get('layout_seed')}, a layout per episode: {bool(r.get('layout_per_episode'))})")
        if key in out:
            raise SystemExit(f"{name}: a second record for {key[0]} episode {key[1]}")
        if key[1] < EPISODES:
            out[key] = r
    return out


class Study:
    """The study's evals, each read once."""

    def __init__(self, layouts: str):
        self.names = {c: name.format(layouts) for c, name in CONTROLLERS.items()}
        self.arms = {c: load(name, layouts) for c, name in self.names.items()}

    def missing(self, controllers: Sequence[str] = tuple(CONTROLLERS)) -> List[str]:
        return [self.names[c] for c in controllers if self.arms[c] is None]


def units(arms: Sequence[Arm]) -> List[int]:
    """The episode indices that every arm has in full: all four tasks."""
    return [e for e in range(EPISODES) if all((t, e) in a for a in arms for t in TASKS)]


def successes(arm: Arm, tasks: Sequence[str], episode: int) -> float:
    return float(sum(bool(arm[t, episode]["success"]) for t in tasks))


def drop(arm: Arm, episode: int) -> float:
    """A unit's drop: the successes on the two trained pairings minus those on the two new pairings (-2 to 2)."""
    return successes(arm, cs.TRAINED, episode) - successes(arm, cs.NEW, episode)


class Result(NamedTuple):
    line: str
    p: float
    mean: float  # of the per-unit values


def paired(values: Sequence[float], episodes: int) -> Result:
    """One value per episode index: the mean as a difference of success rates in points (each side of the difference
    has `episodes` episodes per index), its 95% percentile bootstrap interval over the indices (10,000 draws) and the
    exact two-sided paired sign-flip test."""
    v = np.asarray(values, float)
    lo, hi = bootstrap_interval(v, np.zeros(len(v), int))
    p = sign_flip(v, 1)
    scale = 100 / episodes
    return Result(f"{scale * v.mean():+.1f} points (95% interval {scale * lo:+.1f} to {scale * hi:+.1f}; "
                  f"{v.mean():+.2f} per episode index; {len(v)} episode indices: {int((v > 0).sum())} up, "
                  f"{int((v < 0).sum())} down), exact p = {p:.2g}", p, float(v.mean()))


def run_test(study: Study, controllers: Sequence[str], value, episodes: int) -> Tuple[Optional[Result], List[str]]:
    """A test on the units that all its controllers' evals have in full: value(arms, episode) gives each unit's
    value. (None, what it lacks) when it cannot run."""
    missing = study.missing(controllers)
    arms = [study.arms[c] for c in controllers]
    es = [] if missing else units(arms)
    if not es:
        return None, missing or ["an episode index that every eval has in full"]
    return paired([value(arms, e) for e in es], episodes), []


# The planned tests: (name, question, what the value is, the controllers, the value of a unit).
PLANNED = [
    ("I1", "is the fine-tuned VLA tied to its trained pairings?",
     "VLA-PL's drop (its successes on the two trained pairings minus those on the two new ones)",
     (VLAPL,), lambda a, e: drop(a[0], e)),
    ("I2", "is it tied to them more than the released VLA?", "VLA-PL's drop minus VLA-0's drop",
     (VLAPL, VLA0), lambda a, e: drop(a[0], e) - drop(a[1], e)),
    ("I3", "on a new pairing, does the planner do better than the fine-tuned VLA?",
     "the successes on the two new pairings of the planner reading the VLM's marks, minus VLA-PL's",
     (MARKS, VLAPL), lambda a, e: successes(a[0], cs.NEW, e) - successes(a[1], cs.NEW, e)),
]


def verdict(significant: bool, mean: float) -> str:
    if not significant:
        return "not shown (not significant)"
    return "shown" if mean > 0 else "not shown: significant in the other direction"


def print_tests(study: Study) -> None:
    """I1 to I3, judged together by Holm's method at ALPHA. A test that did not run stays in the family with p = 1,
    so the others' bars are those of three tests."""
    print(f"\n## Planned tests (together by Holm's method at {ALPHA}: the smallest p below {ALPHA / 3:.3g}, the next "
          f"below {ALPHA / 2:.3g}, the last below {ALPHA}; each the exact two-sided paired sign-flip test, one value "
          "per episode index; shown: significant, above 0)")
    results = [run_test(study, controllers, value, len(cs.NEW)) for *_, controllers, value in PLANNED]
    significant = holm([1.0 if r is None else r.p for r, _ in results], ALPHA)
    for (name, question, what, *_), (result, missing), sig in zip(PLANNED, results, significant, strict=True):
        if result is None:
            print(f"- {name}, {question} {what}: not run (it needs {', '.join(missing)}); it counts as p = 1 in "
                  "Holm's method")
        else:
            print(f"- {name}, {question} {what}: {result.line}: {verdict(sig, result.mean)}")


# Secondary results (no claims) ----------------------------------------------------------------------------------


def rate(arm: Arm, tasks: Sequence[str]) -> str:
    ys = [bool(r["success"]) for (t, _), r in arm.items() if t in tasks]
    return f"{100 * np.mean(ys):.1f} ({len(ys)})" if ys else "-"


def print_success(study: Study) -> None:
    """Each controller's success rate on each task, on the trained and on the new pairings, and on all four."""
    print("Success rates, % (episodes; the design's episodes that the eval ran):")
    print("| Controller | " + " | ".join(f"{cs.SHORT[t]} ({'trained' if t in cs.TRAINED else 'new'})" for t in TASKS)
          + " | Trained | New | All |\n" + "|---" * (len(TASKS) + 4) + "|")
    for c, arm in study.arms.items():
        if arm:
            print(f"| {c} | " + " | ".join(rate(arm, [t]) for t in TASKS)
                  + f" | {rate(arm, cs.TRAINED)} | {rate(arm, cs.NEW)} | {rate(arm, TASKS)} |")


def following(arm: Arm) -> Tuple[int, int]:
    """The following score's counts: of the pairs of an episode index and an object whose two tasks the eval ran, how
    many have both tasks solved, and how many pairs there are."""
    pairs = [all(bool(arm[t, e]["success"]) for t in tasks) for e in range(EPISODES)
             for tasks in cs.OBJECT_TASKS.values() if all((t, e) in arm for t in tasks)]
    return int(sum(pairs)), len(pairs)


def print_following(study: Study) -> None:
    print("\nFollowing score: of the pairs of an episode index and an object, the share where the controller solves "
          "both of the object's tasks (a controller that does the same thing whatever the instruction says can solve "
          "only one of the two):")
    for c, arm in study.arms.items():
        both, n = following(arm) if arm else (0, 0)
        if n:
            print(f"- {c}: {100 * both / n:.1f}% ({both} of {n} pairs)")


def logged_name(task: str) -> str:
    """The episode stat of a task's logged pair: its object on the other target."""
    return hs.pair_name(cs.TASKS[task], cs.TASKS[task].logged[0])


def on_trained_target(arm: Arm, task: str) -> Tuple[int, int, int]:
    """A new pairing's episodes that end with the object on its trained target (the logged pair): (how many, of how
    many episodes with the stat, how many records have no such stat)."""
    stats = [r.get("episode_stats", {}).get(logged_name(task)) for (t, _), r in arm.items() if t == task]
    have = [bool(v) for v in stats if v is not None]
    return int(sum(have)), len(have), len(stats) - len(have)


def print_trained_target(study: Study) -> None:
    print("\nOn the new pairings, the episodes that end with the object on its trained target instead (the logged "
          "pair), % (episodes):")
    print("| Controller | " + " | ".join(f"{cs.SHORT[t]}: {logged_name(t)}" for t in cs.NEW) + " | Both |\n"
          + "|---" * (len(cs.NEW) + 2) + "|")
    lacking = {}
    for c, arm in study.arms.items():
        if not arm:
            continue
        counts = [on_trained_target(arm, t) for t in cs.NEW]
        k, n, lacks = (sum(x) for x in zip(*counts, strict=True))
        cells = [f"{100 * a / b:.1f} ({b})" if b else "-" for a, b, _ in [*counts, (k, n, lacks)]]
        print(f"| {c} | " + " | ".join(cells) + " |")
        if lacks:
            lacking[c] = lacks
    for c, lacks in lacking.items():
        print(f"note: {study.names[c]}: {lacks} records of the new pairings have no logged pair in their episode "
              "stats (written before the evaluator logged it); they are left out above")


def print_pairs(study: Study) -> None:
    """The fixed rule against the VLM's marks, and VLA-0's drop alone, with the planned tests' test."""
    print("\nSecondary comparisons (the planned tests' test, no claims):")
    for label, tasks in (("all four tasks", TASKS), ("the trained pairings", cs.TRAINED),
                         ("the new pairings", cs.NEW)):
        result, missing = run_test(study, (MARKS, RULE), lambda a, e, tasks=tasks: successes(a[0], tasks, e)
                                   - successes(a[1], tasks, e), len(tasks))
        print(f"- the {MARKS} minus the {RULE}, {label}: "
              + (result.line if result else f"not run (it needs {', '.join(missing)})"))
    result, missing = run_test(study, (VLA0,), lambda a, e: drop(a[0], e), len(cs.NEW))
    print("- VLA-0's drop (its successes on the two trained pairings minus those on the two new ones): "
          + (result.line if result else f"not run (it needs {', '.join(missing)})"))


def print_checks(study: Study) -> None:
    """A note for each eval with records whose success is not the goal's stat after the last step (they are the same
    check, so they should agree)."""
    goal = {t: hs.pair_name(task, task.goals[0]) for t, task in cs.TASKS.items()}
    for c, arm in study.arms.items():
        differ = [f"{t} episode {e}" for (t, e), r in sorted((arm or {}).items())
                  if r.get("episode_stats", {}).get(goal[t]) not in (None, bool(r["success"]))]
        if differ:
            print(f"note: {study.names[c]}: the success is not the goal's episode stat in {len(differ)} records "
                  f"({', '.join(differ[:5])}{', ...' if len(differ) > 5 else ''})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layouts", default="iid19", help="the layout set of the study's evals")
    a = ap.parse_args()
    study = Study(a.layouts)
    print(f"# Instruction study: the choice scenes on {a.layouts}, episodes 0 to {EPISODES - 1} of each task; one "
          "value per episode index\n")
    if study.missing():
        print(f"Evals that are not there: {', '.join(study.missing())}\n")
    print_success(study)
    print_tests(study)
    print("\n## Secondary (reported, no claims)")
    print_following(study)
    print_trained_target(study)
    print_pairs(study)
    print_checks(study)


if __name__ == "__main__":
    main()
