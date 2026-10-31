"""The marriage study (README.md, "Marriage study", has the plan that this scorer follows).

The planner trains the VLA, and VP-VLA's VLM tells the planner what to move. VLA-0 is the released VP-VLA; the other
models (MODELS) are fine-tuned from it on the planner's demonstrations (VLA-P), with rewordings (VLA-PL), from
layouts that the competence models chose (VLA-PLT), with the planner's corrections (VLA-PLTD), or on VLA-0's own
successes (VLA-S, the control). The planner reads the instruction with its fixed rule or with VP-VLA's VLM marks
(TASK_SOURCE=vp).

    python -m examples.SimplerEnv.eval_files.planner.experiments.score_marriage [--t1 iid16] [--t3 iid17] \\
        [--train iid11]

Test sets: T1, SimplerEnv's 4 tasks with their own instructions (--t1); T2, the 4 reworded sets on SimplerEnv's
layouts; T3, the 5 harder scenes (--t3); T4, T1 at the 3 depth levels (planner-only). vla_name and planner_name give
the run names; the ASMs of the planned test H4 are trained on --train and on the harder scenes' fixed layouts. Each
test takes one value per unit (unit): an episode index of T1 or T3, whose tasks share the layout, or a layout of T2
(one task's episode in the 4 reworded sets). Only the design's episodes count (DESIGN), and of these only the units
that every eval of a test has in full. A test without all its evals is not run, and the evals it lacks are named.
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from examples.SimplerEnv.eval_files import harder_scenes as hs
from examples.SimplerEnv.eval_files.instruction_sets import INSTRUCTION_SETS

from .. import selector as s
from . import runs
from .runs import LEVELS, SHORT
from .score_two_model import print_models
from .stats import cluster_bootstrap_interval, mcnemar, sign_flip, unit_sums

# The models and the prefixes of their VLA-only evals (vla_name); VLA-0's evals have the earlier studies' names.
MODELS = {"VLA-0": None, "VLA-P": "vlap", "VLA-PL": "vlapl", "VLA-PLT": "vlaplt", "VLA-PLTD": "vlapltd",
          "VLA-S": "vlas"}
CURVE = {"VLA-P, 200 demonstrations": "vlap200", "VLA-P, 500 demonstrations": "vlap500"}  # the learning curve
SMALL, SUBTASK = "small policy", "VLA-0 reading the subtask"
PREFIXES = {**MODELS, **CURVE, SMALL: "small", SUBTASK: "vlasub"}
RULE, MARKS = "rule", "vp"  # the planner reads the instruction with its fixed rule, or with VP-VLA's VLM marks
PLANNERS = {RULE: "planner (fixed rule)", MARKS: "planner (VLM marks)"}
SETS = list(INSTRUCTION_SETS)
PARTS = ("T1", "T2", "T3")
DESIGN = {"T1": 100, "T2": 24, "T3": 48}  # the design's episodes of each task (T2: of each task in each set)
CONTROLS, CONTROL_EPISODES = ("blank", "wrong"), 24  # the blank and wrong instructions: T1's first 24 of each task
H4_LEVELS = {"T1": "harsh", "T2": "none", "T3": "none"}  # each test set's depth level in H4
TASK_COLUMNS = [*s.TASKS, *hs.TASKS]
NAMES = {**SHORT, **hs.SHORT}
PLANNER_ONLY, VLA_ONLY, ASM, PER_TASK, ORACLE = "planner-only", "VLA-only", "ASM", "per-task choice", "oracle"

Key = Tuple[str, str, str, int]  # (test set, instruction group, task, episode)
Arm = Dict[Key, float]  # each episode's success


def groups(part: str) -> List[str]:
    """A test set's instruction groups: own (the tasks' own instructions), the reworded sets, or harder."""
    return {"T1": ["own"], "T2": SETS, "T3": ["harder"]}[part]


def group_tasks(group: str) -> List[str]:
    return list(hs.TASKS) if group == "harder" else list(s.TASKS)


def vla_name(prefix: Optional[str], group: str, layouts: str) -> str:
    """A VLA-only eval of an instruction group (own, a reworded set, harder, blank or wrong). VLA-0 (prefix None) has
    the earlier studies' names: vla31_{layouts}, vla_words_{set}_{layouts}, vla_harder_{layouts} and
    vla31_{blank,wrong}_{layouts}; another model's are {prefix}_{layouts}, {prefix}_words_{set}_{layouts},
    {prefix}_harder_{layouts} and {prefix}_{blank,wrong}_{layouts}."""
    if group == "own":
        return f"{prefix or 'vla31'}_{layouts}"
    if group in CONTROLS:
        return f"{prefix or 'vla31'}_{group}_{layouts}"
    return f"{prefix or 'vla'}_{'harder' if group == 'harder' else 'words_' + group}_{layouts}"


def planner_name(source: str, group: str, layouts: str, level: str) -> str:
    """A planner-only eval. The fixed rule: pseed31_{layouts}_{level}, pseed31_harder_{layouts}_{level}, and on a
    reworded set the reworded-instruction study's words_{set}_paper (clean depth). The VLM marks: vp_{layouts}_{level},
    vp_words_{set}_{layouts}_{level} and vp_harder_{layouts}_{level}."""
    if source == RULE and group in SETS:
        if (layouts, level) != ("paper", "none"):
            raise ValueError("the fixed rule's reworded evals are the reworded-instruction study's: paper, clean depth")
        return f"words_{group}_paper"
    middle = "" if group == "own" else "harder_" if group == "harder" else f"words_{group}_"
    return f"{'pseed31' if source == RULE else 'vp'}_{middle}{layouts}_{level}"


def unit(key: Key) -> str:
    """A key's unit: an episode index of T1 or T3 (its tasks share the layout), or a layout of T2 (one task's
    episode, in every reworded set)."""
    part, _, task, episode = key
    return f"{part}/{task}/{episode}" if part == "T2" else f"{part}/{episode}"


def design(part: str) -> Dict[str, List[Key]]:
    """The design's keys of a test set, by unit."""
    out = defaultdict(list)
    for g in groups(part):
        for t in group_tasks(g):
            for e in range(DESIGN[part]):
                out[unit((part, g, t, e))].append((part, g, t, e))
    return out


def complete(arms: Sequence[dict], parts: Sequence[str]) -> List[Key]:
    """The design's keys of the units that every arm has in full."""
    return [k for part in parts for ks in design(part).values() if all(k in a for a in arms for k in ks) for k in ks]


def paired(new: Arm, base: Arm, keys: List[Key]) -> Tuple[str, float, float]:
    """new - base on these keys with one value per unit (its sum): (a line with the mean difference per episode, its
    95% bootstrap interval over the units and the units up and down; the exact sign-flip test's p; the mean)."""
    d = np.array([new[k] - base[k] for k in keys])
    u = np.array([unit(k) for k in keys])
    per = unit_sums(d, u)
    lo, hi = cluster_bootstrap_interval(d, u)
    p = sign_flip(per, 1)
    return (f"{100 * d.mean():+.1f} points (95% interval {100 * lo:+.1f} to {100 * hi:+.1f}; {len(keys)} episodes, "
            f"{len(per)} units: {int((per > 0).sum())} up, {int((per < 0).sum())} down), exact p = {p:.2g}",
            p, float(d.mean()))


def outcomes(records: Dict[Key, dict]) -> Arm:
    return {k: float(r["success"]) for k, r in records.items()}


class Study:
    """The study's evals in $RUNS_DIR, each read once."""

    def __init__(self, t1: str, t3: str, train: str):
        self.layouts, self.train = {"T1": t1, "T2": "paper", "T3": t3}, train
        self.cache: Dict[Tuple[str, str], Optional[dict]] = {}

    def run(self, name: str, group: str, planner: bool) -> Optional[dict]:
        """(task, episode) -> the planner record (planner) or the VLA's success; None when the eval is not there."""
        if (name, group) not in self.cache:
            root = runs.path(name)
            if not os.path.isdir(root):
                self.cache[name, group] = None
            else:
                self.cache[name, group] = s.load_planner(root, group_tasks(group)) if planner else \
                    s.load_vla([root], group_tasks(group))
        return self.cache[name, group]

    def vla(self, prefix: Optional[str], parts: Sequence[str]) -> Tuple[Arm, List[str]]:
        """A VLA-only arm on these test sets, and the evals that are not there."""
        out, missing = {}, []
        for part in parts:
            for g in groups(part):
                name = vla_name(prefix, g, self.layouts[part])
                got = self.run(name, g, planner=False)
                if got is None:
                    missing.append(name)
                else:
                    out.update({(part, g, t, e): y for (t, e), y in got.items()})
        return out, missing

    def planner(self, source: str, parts: Sequence[str], levels: Dict[str, str]) -> Tuple[Dict[Key, dict], List[str]]:
        """A planner-only arm's records on these test sets (each at its level), and the evals that are not there."""
        out, missing = {}, []
        for part in parts:
            for g in groups(part):
                name = planner_name(source, g, self.layouts[part], levels[part])
                got = self.run(name, g, planner=True)
                if got is None:
                    missing.append(name)
                else:
                    out.update({(part, g, t, e): r for (t, e), r in got.items()})
        return out, missing

    def control(self, prefix: Optional[str], group: str) -> Optional[Arm]:
        """A VLA's blank or wrong instructions on T1's first episodes, keyed as T1's own."""
        got = self.run(vla_name(prefix, group, self.layouts["T1"]), group, planner=False)
        return None if got is None else {("T1", "own", t, e): y for (t, e), y in got.items() if e < CONTROL_EPISODES}


# The planned tests ------------------------------------------------------------------------------------------------


class Result(NamedTuple):
    lines: List[str]
    p: float
    win: bool  # significant, with the first one higher (H3: than each of the four)


def pair_test(new: Tuple[Arm, List[str]], base: Tuple[Arm, List[str]],
              parts: Sequence[str]) -> Tuple[Optional[Result], List[str]]:
    missing = new[1] + base[1]
    keys = [] if missing else complete([new[0], base[0]], parts)
    if missing or not keys:
        return None, missing or ["a unit that both evals have in full"]
    line, p, mean = paired(new[0], base[0], keys)
    return Result([line], p, p < 0.05 and mean > 0), []


def h3(study: Study) -> Tuple[Optional[Result], List[str]]:
    """VLA-PL against each option that needs no training on new language, on T2: an intersection-union test."""
    pl, missing = study.vla(MODELS["VLA-PL"], ["T2"])
    others = {}
    for name, (arm, lacks) in {PLANNERS[RULE]: study.planner(RULE, ["T2"], {"T2": "none"}),
                               "VLA-0": study.vla(None, ["T2"]), SUBTASK: study.vla(PREFIXES[SUBTASK], ["T2"]),
                               PLANNERS[MARKS]: study.planner(MARKS, ["T2"], {"T2": "none"})}.items():
        others[name] = outcomes(arm) if name in PLANNERS.values() else arm
        missing += lacks
    keys = [] if missing else complete([pl, *others.values()], ["T2"])
    if missing or not keys:
        return None, missing or ["a unit that every eval has in full"]
    tests = {name: paired(pl, base, keys) for name, base in others.items()}
    p = max(t[1] for t in tests.values())
    win = all(t[1] < 0.05 and t[2] > 0 for t in tests.values())
    return Result([f"vs {name}: {t[0]}" for name, t in tests.items()]
                  + [f"intersection-union: p = {p:.2g} (the largest of the four)"], p, win), []


class Trained(NamedTuple):
    asm: s.TwoModelASM
    rates: Dict[str, Tuple[float, float]]  # each task's planner and VLA success on the episodes the ASM decides


class System(NamedTuple):
    name: str
    source: str  # how its planner reads the instruction
    vla: str  # its VLA, a MODELS key


CURRENT = System("current system (the fixed rule, VLA-0, the ASM)", RULE, "VLA-0")
MARRIED = System("married system (the VLM marks, VLA-PL, the ASM)", MARKS, "VLA-PL")


def train_asm(study: Study, system: System, level: str) -> Tuple[Optional[Trained], List[str]]:
    """The system's ASM at one depth level, with the two-model study's features: trained on the training set (and, at
    clean depth, on the harder scenes' fixed layouts too, for T3's task columns), without the episodes that the VLM
    marks give to the VLA first. Each task's success rates there give the per-task choice."""
    prefix = MODELS[system.vla]
    parts = [(planner_name(system.source, "own", study.train, level), vla_name(prefix, "own", study.train), "own")]
    if level == "none":
        parts.append((planner_name(system.source, "harder", "paper", level), vla_name(prefix, "harder", "paper"),
                      "harder"))
    missing = [n for p, v, _ in parts for n in (p, v) if not os.path.isdir(runs.path(n))]
    if missing:
        return None, missing
    records, tasks, y_plan, y_vla, settings = [], [], [], [], []
    for p, v, g in parts:
        planner, vla, keys = s.matched_layouts(runs.path(p), [runs.path(v)], group_tasks(g))
        settings.append(s.planner_settings(list(planner.values()), runs.path(p)))
        for k in keys:
            if s.vlm_plan_gate(planner[k]) is None:
                records.append(planner[k])
                tasks.append(k[0])
                y_plan.append(float(planner[k]["success"]))
                y_vla.append(vla[k])
    s.merged_settings(settings)  # stops when the parts differ in more than their layouts
    tasks, y_plan, y_vla = np.array(tasks), np.array(y_plan), np.array(y_vla)
    columns = TASK_COLUMNS if level == "none" else list(s.TASKS)
    asm = s.fit_two_model(records, tasks, y_plan, y_vla, task_columns=columns)
    rates = {t: (float(y_plan[tasks == t].mean()), float(y_vla[tasks == t].mean())) for t in columns
             if (tasks == t).any()}
    return Trained(asm, rates), []


class Scored(NamedTuple):
    rows: Dict[str, Arm]  # planner-only, VLA-only, the ASM, the per-task choice, the oracle
    probs: Dict[Key, Tuple[float, float]]  # P(planner), P(VLA) of each episode that the ASM decides


def score_system(study: Study, system: System) -> Tuple[Optional[Scored], List[str]]:
    """The system on H4's test sets, each at its level (H4_LEVELS), scored offline: each episode's choice from the
    first look of its planner-only eval (the gate, then the ASM), and the chosen method's outcome."""
    trained, missing = {}, []
    for level in sorted(set(H4_LEVELS.values())):
        trained[level], lacks = train_asm(study, system, level)
        missing += lacks
    records, lacks = study.planner(system.source, PARTS, H4_LEVELS)
    vla, lacks_vla = study.vla(MODELS[system.vla], PARTS)
    missing += lacks + lacks_vla
    if missing:
        return None, missing
    rows, probs = {n: {} for n in (PLANNER_ONLY, VLA_ONLY, ASM, PER_TASK, ORACLE)}, {}
    for part in PARTS:
        t = trained[H4_LEVELS[part]]
        keys = [k for k in records if k[0] == part and k in vla]
        if not keys:
            continue
        recs = [records[k] for k in keys]
        p_plan, p_vla = t.asm.probs(recs, [k[2] for k in keys])
        for k, r, pp, pv in zip(keys, recs, p_plan, p_vla, strict=True):
            y_plan, y_vla, gated = float(r["success"]), vla[k], s.vlm_plan_gate(r) is not None
            plan_rate, vla_rate = t.rates.get(k[2], (1.0, 0.0))
            rows[PLANNER_ONLY][k], rows[VLA_ONLY][k], rows[ORACLE][k] = y_plan, y_vla, max(y_plan, y_vla)
            rows[ASM][k] = y_vla if gated or pp < pv else y_plan
            rows[PER_TASK][k] = y_vla if gated or plan_rate < vla_rate else y_plan
            if not gated:
                probs[k] = (float(pp), float(pv))
    return Scored(rows, probs), []


def h4(systems: Dict[System, Tuple[Optional[Scored], List[str]]]) -> Tuple[Optional[Result], List[str]]:
    (married, m1), (current, m2) = systems[MARRIED], systems[CURRENT]
    if married is None or current is None:
        return None, m1 + m2
    return pair_test((married.rows[ASM], []), (current.rows[ASM], []), PARTS)


def print_tests(study: Study, systems: Dict[System, Tuple[Optional[Scored], List[str]]]) -> None:
    """H1 to H5 in their fixed order; a test counts only when every test before it was a win."""
    tests = [
        ("H1", "does the planner teach the VLA? VLA-P minus VLA-0 on T1",
         lambda: pair_test(study.vla(MODELS["VLA-P"], ["T1"]), study.vla(None, ["T1"]), ["T1"])),
        ("H2", "is it the planner's data? VLA-PL minus VLA-S on T1 and T2",
         lambda: pair_test(study.vla(MODELS["VLA-PL"], ["T1", "T2"]), study.vla(MODELS["VLA-S"], ["T1", "T2"]),
                           ["T1", "T2"])),
        ("H3", "does the married VLA beat every option that needs no training on new language? VLA-PL minus each, "
               "on T2", lambda: h3(study)),
        ("H4", f"does the married system beat the current one? The {MARRIED.name} minus the {CURRENT.name}, on T1 "
               "(harsh depth), T2 and T3", lambda: h4(systems)),
        ("H5", "does choosing where to collect help? VLA-PLT minus VLA-PL on T1 and T2",
         lambda: pair_test(study.vla(MODELS["VLA-PLT"], ["T1", "T2"]), study.vla(MODELS["VLA-PL"], ["T1", "T2"]),
                           ["T1", "T2"])),
    ]
    print("\n## Planned tests (fixed order; each the exact two-sided paired sign-flip test at 0.05, one value per "
          "unit)")
    counts = True
    for name, question, run in tests:
        result, missing = run()
        if result is None:
            print(f"{name}, {question}: not run (it needs {', '.join(missing)})")
            counts = False
            continue
        word = (": a win" if result.win else ": not a win") if counts else \
            ": does not count (a test before it was not a win or did not run)"
        if len(result.lines) == 1:
            print(f"{name}, {question}: {result.lines[0]}{word}")
        else:
            print(f"{name}, {question}:\n" + "\n".join(f"  - {line}" for line in result.lines[:-1])
                  + f"\n  {result.lines[-1]}{word}")
        counts = counts and result.win


# Secondary results (no claims) ----------------------------------------------------------------------------------


def rate(arm: Arm, part: str, group: Optional[str] = None) -> str:
    ys = [y for k, y in arm.items() if k[0] == part and k[1] == (group or k[1]) and k[3] < DESIGN[part]]
    return f"{100 * np.mean(ys):.1f} ({len(ys)})" if ys else "-"


def print_success(study: Study) -> None:
    """Every eval's success rate (episodes) on T1, each reworded set, all of T2, and T3."""
    print("Success rates, % (episodes; the design's episodes that the eval ran):")
    print("| Eval | T1 | " + " | ".join(SETS) + " | T2 | T3 |\n" + "|---" * (len(SETS) + 4) + "|")
    rows = [(name, study.vla(prefix, PARTS)[0]) for name, prefix in PREFIXES.items()]
    for source, label in PLANNERS.items():
        for level in LEVELS:
            parts = PARTS if level == "none" else ("T1",)
            rows.append((f"{label}, {runs.LEVEL_NAMES[level]} depth",
                         outcomes(study.planner(source, parts, {p: level for p in parts})[0])))
    for name, arm in rows:
        if arm:
            print(f"| {name} | {rate(arm, 'T1')} | " + " | ".join(rate(arm, "T2", g) for g in SETS)
                  + f" | {rate(arm, 'T2')} | {rate(arm, 'T3')} |")


def print_vs_vla0(study: Study) -> None:
    base = study.vla(None, PARTS)[0]
    cells = [("T1", "own"), *[("T2", g) for g in SETS], ("T3", "harder")]
    print("\nEach model against VLA-0 (exact McNemar on the design's episodes that both ran: better / worse, p):")
    print("| Model | T1 | " + " | ".join(SETS) + " | T3 |\n" + "|---" * (len(SETS) + 3) + "|")
    for name, prefix in PREFIXES.items():
        arm = study.vla(prefix, PARTS)[0]
        if prefix is None or not arm:
            continue
        out = []
        for part, g in cells:
            keys = [k for k in arm if k[:2] == (part, g) and k in base and k[3] < DESIGN[part]]
            if not keys:
                out.append("-")
                continue
            b, w, p = mcnemar(np.array([base[k] for k in keys]), np.array([arm[k] for k in keys]))
            out.append(f"{b} / {w}, p = {p:.2g}")
        print(f"| {name} | " + " | ".join(out) + " |")


def print_pair(label: str, new: Tuple[Arm, List[str]], base: Tuple[Arm, List[str]], parts: Sequence[str]) -> None:
    result, missing = pair_test(new, base, parts)
    print(f"- {label}: " + (result.lines[0] if result else f"not run (it needs {', '.join(missing)})"))


def print_pairs(study: Study) -> None:
    """The secondary comparisons with the planned tests' test (no claims)."""
    t12 = ["T1", "T2"]
    print("\nSecondary comparisons (the planned tests' test, no claims):")
    print_pair("VLA-PLTD minus VLA-PLT, T1 and T2", study.vla(MODELS["VLA-PLTD"], t12),
               study.vla(MODELS["VLA-PLT"], t12), t12)
    for level in LEVELS:  # T4: the student against its teacher
        teacher = study.planner(RULE, ["T1"], {"T1": level})
        teacher = (outcomes(teacher[0]), teacher[1])
        for model in ("VLA-PL", "VLA-0"):
            print_pair(f"{model} minus the {PLANNERS[RULE]}, T1 with {runs.LEVEL_NAMES[level]} depth",
                       study.vla(MODELS[model], ["T1"]), teacher, ["T1"])
    for part, levels in (("T1", LEVELS), ("T2", ["none"]), ("T3", ["none"])):
        for level in levels:
            marks, rule = (study.planner(src, [part], {part: level}) for src in (MARKS, RULE))
            print_pair(f"{PLANNERS[MARKS]} minus the {PLANNERS[RULE]}, {part} with {runs.LEVEL_NAMES[level]} depth",
                       (outcomes(marks[0]), marks[1]), (outcomes(rule[0]), rule[1]), [part])
    print_pair(f"VLA-P minus the {SMALL} (the same demonstrations), T1", study.vla(MODELS["VLA-P"], ["T1"]),
               study.vla(PREFIXES[SMALL], ["T1"]), ["T1"])
    for name in [*CURVE, "VLA-P"]:  # the learning curve: 200, 500 and 1,000 demonstrations
        print_pair(f"{name} minus VLA-0, T1", study.vla(PREFIXES[name], ["T1"]), study.vla(None, ["T1"]), ["T1"])


def print_solved(study: Study) -> None:
    """Of T1's episodes that the planner (fixed rule, clean depth) solves and VLA-0 does not, the share each model
    solves."""
    planner = outcomes(study.planner(RULE, ["T1"], {"T1": "none"})[0])
    vla0 = study.vla(None, ["T1"])[0]
    only = [k for k in planner if k in vla0 and planner[k] == 1 and vla0[k] == 0 and k[3] < DESIGN["T1"]]
    print(f"\nT1 episodes that only the {PLANNERS[RULE]} (clean depth) solves, not VLA-0: {len(only)}")
    for name, prefix in MODELS.items():
        arm = study.vla(prefix, ["T1"])[0]
        have = [k for k in only if k in arm]
        if prefix and have:
            print(f"- {name} solves {sum(arm[k] for k in have):.0f} of the {len(have)} that it ran")


def print_systems(systems: Dict[System, Tuple[Optional[Scored], List[str]]]) -> None:
    """H4's systems on each test set (secondary)."""
    (married, _), (current, _) = systems[MARRIED], systems[CURRENT]
    if married is None or current is None:
        print("\nH4's systems: not scored (their evals are not all there)")
        return
    old = current.rows[ORACLE]  # the old pair's oracle: the better of the fixed rule and VLA-0 in each episode
    for part in PARTS:
        keys = complete([married.rows[ASM], current.rows[ASM]], [part])
        if not keys:
            continue
        print(f"\n### H4's systems on {part} ({runs.LEVEL_NAMES[H4_LEVELS[part]]} depth; {len(keys)} episodes)")
        print("| System | Row | " + " | ".join(groups(part)) + " | All |\n|---|---|" + "---|" * (len(groups(part)) + 1))
        for system, scored in ((CURRENT, current), (MARRIED, married)):
            for row, arm in scored.rows.items():
                print(f"| {system.name} | {row} | " + " | ".join(
                    f"{100 * np.mean([arm[k] for k in keys if k[1] == g]):.1f}" for g in groups(part))
                    + f" | {100 * np.mean([arm[k] for k in keys]):.1f} |")
        for system, scored in ((CURRENT, current), (MARRIED, married)):
            p, v = (np.array([scored.rows[n][k] for k in keys]) for n in (PLANNER_ONLY, VLA_ONLY))
            print(f"- {system.name}: both methods solve {int(((p == 1) & (v == 1)).sum())}, only the planner "
                  f"{int(((p == 1) & (v == 0)).sum())}, only the VLA {int(((p == 0) & (v == 1)).sum())}, neither "
                  f"{int(((p == 0) & (v == 0)).sum())}")
        print(f"- the married system minus the old pair's oracle: {paired(married.rows[ASM], old, keys)[0]}")
        if part == "T2":
            for g in SETS:
                ks = [k for k in keys if k[1] == g]
                print(f"  - {g}: the married system {100 * np.mean([married.rows[ASM][k] for k in ks]):.1f}, the old "
                      f"pair's oracle {100 * np.mean([old[k] for k in ks]):.1f}")
        for system, scored in ((CURRENT, current), (MARRIED, married)):
            decided = [k for k in keys if k in scored.probs]
            if decided:
                print(f"\n{system.name}, the {len(decided)} episodes that its ASM decides:")
                probs = np.array([scored.probs[k] for k in decided])
                print_models({"P(planner)": (probs[:, 0], np.array([scored.rows[PLANNER_ONLY][k] for k in decided])),
                              "P(VLA)": (probs[:, 1], np.array([scored.rows[VLA_ONLY][k] for k in decided]))},
                             np.array([k[2] for k in decided]), bins=False)


def print_controls(study: Study) -> None:
    """Does a VLA read the instruction: its success with an empty and a wrong instruction against its own, on T1's
    first episodes (exact McNemar)."""
    print(f"\nBlank and wrong instructions (T1's first {CONTROL_EPISODES} episodes of each task; exact McNemar against "
          "the own instruction: better / worse, p):")
    for name in ("VLA-0", "VLA-PL"):
        own = study.vla(MODELS[name], ["T1"])[0]
        for group in CONTROLS:
            arm = study.control(MODELS[name], group)
            keys = [k for k in (arm or {}) if k in own]
            if not keys:
                print(f"- {name}, {group}: not run")
                continue
            b, w, p = mcnemar(np.array([own[k] for k in keys]), np.array([arm[k] for k in keys]))
            print(f"- {name}, {group}: {100 * np.mean([arm[k] for k in keys]):.1f} against "
                  f"{100 * np.mean([own[k] for k in keys]):.1f} with the own instruction ({len(keys)} episodes; "
                  f"{b} / {w}, p = {p:.2g})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t1", default="iid16", help="T1's layouts (SimplerEnv's tasks with their own instructions)")
    ap.add_argument("--t3", default="iid17", help="T3's layouts (the harder scenes)")
    ap.add_argument("--train", default="iid11", help="the training set of H4's ASMs")
    a = ap.parse_args()
    study = Study(a.t1, a.t3, a.train)
    print(f"# Marriage study: T1 {a.t1}, T2 the reworded sets on paper, T3 the harder scenes on {a.t3}; H4's ASMs "
          f"trained on {a.train} and the harder scenes' fixed layouts\n")
    print_success(study)
    systems = {system: score_system(study, system) for system in (CURRENT, MARRIED)}
    print_tests(study, systems)
    print("\n## Secondary (reported, no claims)")
    print_vs_vla0(study)
    print_pairs(study)
    print_solved(study)
    print_systems(systems)
    print_controls(study)


if __name__ == "__main__":
    main()
