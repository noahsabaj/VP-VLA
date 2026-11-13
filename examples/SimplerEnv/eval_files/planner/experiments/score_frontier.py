"""The frontier-model study (README.md, "Frontier-model study", has the plan that this scorer follows).

A general frontier model in the system, in two arms. The plan arm: the frontier model answers the VLM plan's
question, and the planner does the motion (the evals fplan_*). The agent arm: the frontier model drives the arm
itself with tools (the evals fagent_*).

    python -m examples.SimplerEnv.eval_files.planner.experiments.score_frontier [--t1 iid16] [--t3 iid17] \\
        [--t5 iid18] [--train iid11] [--answers FILE ...]

Rows (ROWS): the planner reading the frontier plan (fplan_*), the planner reading Qwen3-VL's plan (vlmplan_*), the
frontier agent (fagent_*), the planner reading the VLM's marks (vp_*), VLA-PL (vlapl_*), and the married system,
which score_marriage's rules score offline (H4's on T2 and T3, H5's on T5). plan_name and agent_name give the new
evals' names; the others' are the marriage study's. All with clean depth.

The planned tests take T2, T3 and T5 together, with the marriage study's units (score_marriage.unit), its design
(DESIGN: only the design's episodes count) and its test; of the units, only those that every eval of a test has in
full. A test without all its evals is not run, and the evals it lacks are named. The agent arm's tests take the test
sets that the agent ran: those with all of their agent evals in $RUNS_DIR.

How this scorer reads the plan's sentence "a significant difference in the other direction is reported as a loss":
a test is a loss when it is significant with the frontier arm lower. F1 has two comparisons and its p is the larger
one, so F1 is a win when the frontier plan is significantly higher than each of the two, and a loss when it is
significantly lower than each of the two; each comparison's own line gives its direction.

--answers: the plan arm's answers files (frontier_answers.py), for the refusals, errors, tokens, cost and time
(default: every .jsonl file in $RUNS_DIR/fanswers).
"""

import argparse
import glob
import json
import os
from collections import Counter
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from .. import frontier
from . import frontier_answers, runs
from . import score_marriage as sm
from .score_marriage import Arm, Key

FPLAN, QPLAN, AGENT, MARKS, VLAPL, MARRIED = ("frontier plan", "Qwen3-VL's plan", "frontier agent", "VLM's marks",
                                              "VLA-PL", "married system")
ROWS = (FPLAN, QPLAN, AGENT, MARKS, VLAPL, MARRIED)
THE = {FPLAN: "the frontier plan", QPLAN: "Qwen3-VL's plan", AGENT: "the frontier agent", MARKS: "the VLM's marks",
       VLAPL: "VLA-PL", MARRIED: "the married system"}  # each row in a sentence
PLAN_PREFIX = {FPLAN: "fplan", QPLAN: "vlmplan"}  # the planner-only evals with TASK_SOURCE=vlm
AGENT_PREFIX = "fagent"
TEST_PARTS = ("T2", "T3", "T5")  # the planned tests' test sets
AGENT_ORDER = ("T5", "T2", "T3")  # the order in which the agent arm runs whole sets
MARRIED_RULES = (("T2", "T3"), ("T5",))  # the married system's rule on each test set: H4's, H5's
ALPHA = 0.025  # each family's level: Bonferroni for the two families, 0.05 for the study
CLEAN, HARSH = "none", "harsh"

Got = Tuple[Arm, List[str]]  # a row's outcomes, and the evals that are not there


def middle(group: str) -> str:
    return "" if group == "own" else f"{group}_" if group in ("harder", "novel") else f"words_{group}_"


def plan_name(prefix: str, group: str, layouts: str, level: str) -> str:
    """A planner-only eval with the VLM plan: {prefix}_{layouts}_{level}, {prefix}_words_{set}_{layouts}_{level},
    {prefix}_harder_{layouts}_{level} and {prefix}_novel_{layouts}_{level}; prefix fplan (the frontier model's
    answers) or vlmplan (Qwen3-VL's)."""
    return f"{prefix}_{middle(group)}{layouts}_{level}"


def agent_name(group: str, layouts: str) -> str:
    """An eval of the frontier agent: fagent_words_{set}_{layouts}, fagent_harder_{layouts}, fagent_novel_{layouts}."""
    return f"{AGENT_PREFIX}_{middle(group)}{layouts}"


def load_agent(root: str, tasks: Sequence[str]) -> Dict[Tuple[str, int], dict]:
    """(task, episode) -> the agent's record (frontier.AGENT_FILE in each task's folder, and in its chunks' folders).
    A second record for an episode, or an end that is not one of frontier.AGENT_ENDS, stops the load."""
    out = {}
    paths = [p for task in tasks
             for p in sorted(glob.glob(os.path.join(root, task, "**", frontier.AGENT_FILE), recursive=True))]
    for path in paths:
        task = os.path.relpath(path, root).split(os.sep)[0]
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                r = json.loads(line)
                key = (task, int(r["obj_episode_id"]))
                if key in out:
                    raise SystemExit(f"{path}:{n}: a second record for {task} episode {key[1]}")
                if r.get("end") not in frontier.AGENT_ENDS:
                    raise SystemExit(f"{path}:{n}: the end {r.get('end')!r} is not one of {frontier.AGENT_ENDS}")
                out[key] = r
    return out


class Frontier:
    """The study's evals in $RUNS_DIR, each read once."""

    def __init__(self, t1: str = "iid16", t3: str = "iid17", t5: str = "iid18", train: str = "iid11"):
        self.study = sm.Study(t1, t3, train, t5)
        self.layouts = self.study.layouts
        self.cache: Dict[tuple, tuple] = {}

    def plan(self, row: str, parts: Sequence[str], level: str = CLEAN) -> Tuple[Dict[Key, dict], List[str]]:
        """The records of a planner-only eval with the VLM plan (FPLAN or QPLAN) on these test sets, and the evals
        that are not there. A record of another depth level or of another task source, a frontier plan's record
        without a saved answer, or a Qwen3-VL record with one, stops the load: it is not this row's eval."""
        out, missing = {}, []
        for part in parts:
            for g in sm.groups(part):
                name = plan_name(PLAN_PREFIX[row], g, self.layouts[part], level)
                got = self.study.run(name, g, planner=True)
                if got is None:
                    missing.append(name)
                    continue
                for (t, e), r in got.items():
                    saved = r.get("vlm_plan_answer") is not None
                    if r.get("task_source") != "vlm" or r.get("depth_noise") != level or saved != (row == FPLAN):
                        raise SystemExit(
                            f"{name}: {t} episode {e} is not an episode of {THE[row]} at {level} depth (task source "
                            f"{r.get('task_source')!r}, depth {r.get('depth_noise')!r}, a saved answer: {saved})")
                    out[part, g, t, e] = r
        return out, missing

    def agent_records(self, parts: Sequence[str]) -> Tuple[Dict[Key, dict], List[str]]:
        out, missing = {}, []
        for part in parts:
            for g in sm.groups(part):
                name = agent_name(g, self.layouts[part])
                if ("agent", name) not in self.cache:
                    root = runs.path(name)
                    self.cache["agent", name] = (load_agent(root, sm.group_tasks(g)) if os.path.isdir(root) else None,)
                got = self.cache["agent", name][0]
                if got is None:
                    missing.append(name)
                else:
                    out.update({(part, g, t, e): r for (t, e), r in got.items()})
        return out, missing

    def married(self, parts: Sequence[str]) -> Got:
        """The married system's outcomes on these test sets, scored offline by the marriage study's rules: each
        episode's choice from the first look of the VLM marks' planner-only eval (the gate, then the ASM), and the
        chosen method's outcome."""
        out, missing = {}, []
        for rule in MARRIED_RULES:
            mine = tuple(p for p in rule if p in parts)
            if not mine:
                continue
            if ("married", mine) not in self.cache:
                self.cache["married", mine] = sm.score_system(self.study, sm.MARRIED, mine)
            scored, lacks = self.cache["married", mine]
            missing += [n for n in lacks if n not in missing]
            if scored is not None:
                out.update(scored.rows[sm.ASM])
        return out, missing

    def row(self, row: str, parts: Sequence[str]) -> Got:
        """A row's outcomes on these test sets (clean depth), and the evals that are not there."""
        parts = [p for p in parts if not (p == "T1" and row in (AGENT, MARRIED))]  # no such eval of T1
        if row in PLAN_PREFIX:
            records, missing = self.plan(row, parts)
        elif row == AGENT:
            records, missing = self.agent_records(parts)
        elif row == MARKS:
            records, missing = self.study.planner(sm.MARKS, parts, dict.fromkeys(parts, CLEAN))
        elif row == VLAPL:
            return self.study.vla(sm.MODELS["VLA-PL"], parts)
        else:
            return self.married(parts)
        return sm.outcomes(records), missing

    def agent_parts(self) -> Tuple[List[str], List[str]]:
        """(the test sets that the agent ran: all of their agent evals are there; notes on sets with some of them)."""
        ran, notes = [], []
        for part in AGENT_ORDER:
            names = [agent_name(g, self.layouts[part]) for g in sm.groups(part)]
            there = [n for n in names if os.path.isdir(runs.path(n))]
            if len(there) == len(names):
                ran.append(part)
            elif there:
                notes.append(f"{part} is left out of the agent arm's tests: it has {', '.join(there)} but not "
                             f"{', '.join(n for n in names if n not in there)}")
        return ran, notes


# The planned tests ------------------------------------------------------------------------------------------------


class Result(NamedTuple):
    lines: List[str]
    p: float  # of an intersection-union test: the largest of its comparisons'
    win: bool  # significant, with the frontier arm higher (than each)
    loss: bool  # significant, with the frontier arm lower (than each)


def compare(new: Got, bases: Dict[str, Got], parts: Sequence[str]) -> Tuple[Optional[Result], List[str]]:
    """new minus each base on these test sets, on the units that all of them have in full. With more than one base
    it is an intersection-union test: its p is the largest one. (None, what it lacks) when it cannot run."""
    missing = list(new[1])
    for _, lacks in bases.values():
        missing += [n for n in lacks if n not in missing]
    if not parts:
        return None, missing or ["a test set that the agent ran"]
    keys = [] if missing else sm.complete([new[0], *(arm for arm, _ in bases.values())], parts)
    if missing or not keys:
        return None, missing or ["a unit that every eval has in full"]
    tests = {name: sm.paired(new[0], arm, keys) for name, (arm, _) in bases.items()}
    p = max(t[1] for t in tests.values())
    win = all(t[1] < ALPHA and t[2] > 0 for t in tests.values())
    loss = all(t[1] < ALPHA and t[2] < 0 for t in tests.values())
    if len(tests) == 1:
        return Result([t[0] for t in tests.values()], p, win, loss), []
    return Result([f"vs {THE[name]}: {t[0]}" for name, t in tests.items()]
                  + [f"intersection-union: p = {p:.2g} (the largest of the {len(tests)})"], p, win, loss), []


def verdict(result: Result, counts: bool) -> str:
    if not counts:
        return "does not count (a test before it in its family was not a win or did not run)"
    if result.win:
        return "a win"
    if result.loss:
        return "a loss (significant, with the frontier arm lower)"
    return "not a win (not significant)" if result.p >= ALPHA else \
        "not a win (significant comparisons in different directions)"


def families(fr: Frontier) -> Dict[str, Tuple[Sequence[str], list]]:
    """Each family's test sets and its tests in their fixed order: (name, question, what is compared, the test)."""
    ran, _ = fr.agent_parts()
    plan = [
        ("F1", "is a frontier model a better front-end for the planner than the 4B VLM?",
         f"{THE[FPLAN]} minus each of {THE[QPLAN]} and {THE[MARKS]}",
         lambda: compare(fr.row(FPLAN, TEST_PARTS), {QPLAN: fr.row(QPLAN, TEST_PARTS),
                                                     MARKS: fr.row(MARKS, TEST_PARTS)}, TEST_PARTS)),
        ("F2", "does the frontier model with the planner beat the trained VLA?", f"{THE[FPLAN]} minus {THE[VLAPL]}",
         lambda: compare(fr.row(FPLAN, TEST_PARTS), {VLAPL: fr.row(VLAPL, TEST_PARTS)}, TEST_PARTS)),
        ("F3", "does it beat the married system?", f"{THE[FPLAN]} minus {THE[MARRIED]}",
         lambda: compare(fr.row(FPLAN, TEST_PARTS), {MARRIED: fr.row(MARRIED, TEST_PARTS)}, TEST_PARTS)),
    ]
    agent = [
        ("A1", "does the frontier model need the planner?", f"{THE[AGENT]} minus {THE[FPLAN]}",
         lambda: compare(fr.row(AGENT, ran), {FPLAN: fr.row(FPLAN, ran)}, ran)),
        ("A2", "does the agent beat the trained VLA?", f"{THE[AGENT]} minus {THE[VLAPL]}",
         lambda: compare(fr.row(AGENT, ran), {VLAPL: fr.row(VLAPL, ran)}, ran)),
        ("A3", "does it beat the married system?", f"{THE[AGENT]} minus {THE[MARRIED]}",
         lambda: compare(fr.row(AGENT, ran), {MARRIED: fr.row(MARRIED, ran)}, ran)),
    ]
    return {"The plan arm": (TEST_PARTS, plan), "The agent arm": (ran, agent)}


def run_family(tests: list) -> List[Tuple[str, Optional[Result], List[str], bool]]:
    """Each test of a family in its fixed order: (name, the result or None, what it lacks, whether it counts). A
    test counts only when every test before it was a win."""
    out, counts = [], True
    for name, _, _, run in tests:
        result, missing = run()
        out.append((name, result, missing, counts and result is not None))
        counts = counts and result is not None and result.win
    return out


def print_tests(fr: Frontier) -> None:
    print(f"\n## Planned tests (two families, each at {ALPHA}; in a family the order is fixed and a test counts only "
          "when every test before it was a win; each the exact two-sided paired sign-flip test, one value per unit)")
    _, notes = fr.agent_parts()
    for family, (parts, tests) in families(fr).items():
        print(f"\n### {family}: " + (" + ".join(sorted(parts)) if parts else "no test set ran"))
        for (name, question, what, _), (_, result, missing, counts) in zip(tests, run_family(tests), strict=True):
            if result is None:
                print(f"{name}, {question} {what[0].upper()}{what[1:]}: not run (it needs {', '.join(missing)})")
            elif len(result.lines) == 1:
                print(f"{name}, {question} {what[0].upper()}{what[1:]}: {result.lines[0]}: "
                      f"{verdict(result, counts)}")
            else:
                print(f"{name}, {question} {what[0].upper()}{what[1:]}:\n"
                      + "\n".join(f"  - {line}" for line in result.lines[:-1])
                      + f"\n  {result.lines[-1]}: {verdict(result, counts)}")
    for note in notes:
        print(f"note: {note}")


# Secondary results (no claims) ----------------------------------------------------------------------------------


def rate(arm: Arm, keep: Callable[[Key], bool]) -> str:
    """The success rate in % (episodes) of the design's episodes that the row has and `keep` takes."""
    ys = [y for k, y in arm.items() if k[3] < sm.DESIGN[k[0]] and keep(k)]
    return f"{100 * np.mean(ys):.1f} ({len(ys)})" if ys else "-"


def print_success(fr: Frontier) -> None:
    """Every row's success rate on each test set, each reworded set and each task."""
    parts = ("T1", *TEST_PARTS)
    arms = {row: fr.row(row, parts)[0] for row in ROWS}
    print("Success rates, % (episodes; the design's episodes that the row's evals ran; clean depth):")
    print("| Row | T1 (secondary) | T2 | T3 | T5 | T2 + T3 + T5 |\n|---|---|---|---|---|---|")
    for row, arm in arms.items():
        print(f"| {row} | " + " | ".join(rate(arm, lambda k, p=p: k[0] == p) for p in parts)
              + f" | {rate(arm, lambda k: k[0] in TEST_PARTS)} |")
    print("\nT2 by reworded set:\n| Row | " + " | ".join(sm.SETS) + " |\n" + "|---" * (len(sm.SETS) + 1) + "|")
    for row, arm in arms.items():
        print(f"| {row} | " + " | ".join(rate(arm, lambda k, g=g: k[0] == "T2" and k[1] == g) for g in sm.SETS)
              + " |")
    for part in parts:
        if not any(k[0] == part for arm in arms.values() for k in arm):
            continue
        tasks = sm.group_tasks(sm.groups(part)[0])
        print(f"\n{part} by task:\n| Row | " + " | ".join(sm.NAMES[t] for t in tasks) + " |\n"
              + "|---" * (len(tasks) + 1) + "|")
        for row, arm in arms.items():
            if any(k[0] == part for k in arm):
                print(f"| {row} | " + " | ".join(rate(arm, lambda k, t=t, p=part: k[0] == p and k[2] == t)
                                                 for t in tasks) + " |")


def print_t1(fr: Frontier) -> None:
    """The frontier plan on T1 with clean and with harsh depth: the answers are the same, only the planner's depth
    changes."""
    got = {level: fr.plan(FPLAN, ["T1"], level) for level in (CLEAN, HARSH)}
    print("\nThe frontier plan on T1 (secondary), clean and harsh depth:")
    for level, (records, missing) in got.items():
        print(f"- {runs.LEVEL_NAMES[level]} depth: "
              + (f"not there ({', '.join(missing)})" if missing
                 else f"{rate(sm.outcomes(records), lambda k: True)} % (episodes)"))
    (clean, m1), (harsh, m2) = got[CLEAN], got[HARSH]
    keys = [] if m1 or m2 else sm.complete([clean, harsh], ["T1"])
    if keys:
        print(f"- harsh minus clean: {sm.paired(sm.outcomes(harsh), sm.outcomes(clean), keys)[0]}")
        differ = sum(clean[k].get("vlm_plan_text") != harsh[k].get("vlm_plan_text") for k in keys)
        print(f"- episodes whose answer is not the same at the two depth levels: {differ} of {len(keys)}"
              + ("" if not differ else " (they should be the same: the picture is the same)"))


def print_plans(fr: Frontier) -> None:
    """How often each plan says "not feasible", and how often its point lies in the SAM mask that the planner uses
    (of the names for which SAM gave a mask)."""
    print('\nThe plans: "not feasible", and the points in the SAM mask that the planner uses (of the names that SAM '
          "gave a mask for), % (count):")
    print("| Row | Test set | Episodes | Not feasible | of these: refusals, errors | Names | Point in the mask |\n"
          "|---|---|---|---|---|---|---|")
    notes = []
    for row in PLAN_PREFIX:
        records, _ = fr.plan(row, ("T1", *TEST_PARTS))
        models = Counter((r.get("vlm_plan_answer") or {}).get("model") for r in records.values()
                         if (r.get("vlm_plan_answer") or {}).get("model"))
        for part in ("T1", *TEST_PARTS):
            rs = [r for k, r in records.items() if k[0] == part and k[3] < sm.DESIGN[part]]
            if not rs:
                continue
            no = [r for r in rs if not r.get("vlm_plan_feasible")]
            errors = Counter("refusal" if (r.get("vlm_plan_answer") or {}).get("error") == "refusal" else "error"
                             for r in no if (r.get("vlm_plan_answer") or {}).get("error"))
            names = [n for r in rs for n in r.get("vlm_plan_sam", [])]
            inside = sum(bool(n.get("covers_point")) for n in names)
            print(f"| {row} | {part} | {len(rs)} | {100 * len(no) / len(rs):.1f} ({len(no)}) | "
                  f"{errors['refusal']}, {errors['error']} | {len(names)} | "
                  + (f"{100 * inside / len(names):.1f} ({inside})" if names else "-") + " |")
        if models:
            notes.append(f"note: the {row}'s saved answers came from: "
                         + ", ".join(f"{m} ({n} episodes)" for m, n in models.most_common()))
    for note in notes:
        print(note)


def print_agent(fr: Frontier) -> None:
    """The agent's turns per episode, how its episodes end and the success rate of each end, and its refusals,
    errors, tokens, cost and time per episode."""
    ran, _ = fr.agent_parts()
    records, _ = fr.agent_records(ran)
    rs = [r for k, r in records.items() if k[3] < sm.DESIGN[k[0]]]
    if not rs:
        print("\nThe frontier agent: no eval yet")
        return
    turns = np.array([r["turns"] for r in rs])
    print(f"\nThe frontier agent ({len(rs)} episodes of {' + '.join(sorted(ran))}): {turns.mean():.1f} turns per "
          f"episode (median {np.median(turns):.0f}, most {turns.max()})")
    print("| End | Episodes, % (count) | Success, % |\n|---|---|---|")
    for end in frontier.AGENT_ENDS:
        mine = [r for r in rs if r["end"] == end]
        print(f"| {end} | {100 * len(mine) / len(rs):.1f} ({len(mine)}) | "
              + (f"{100 * np.mean([bool(r['success']) for r in mine]):.1f}" if mine else "-") + " |")
    usage = Counter()
    for r in rs:
        usage.update({k: v or 0 for k, v in (r.get("usage") or {}).items()})
    costs = [r["cost_usd"] for r in rs if r.get("cost_usd") is not None]
    seconds = [r["model_seconds"] for r in rs if r.get("model_seconds") is not None]
    print(f"- refusals: {sum(bool(r.get('refusal')) for r in rs)}; errors: {sum(bool(r.get('error')) for r in rs)}")
    print("- tokens per episode: " + (", ".join(f"{k} {v / len(rs):,.0f}" for k, v in sorted(usage.items())) or "-"))
    print("- cost: " + (f"{sum(costs):.2f} USD ({np.mean(costs):.3f} per episode, {len(costs)} episodes)" if costs
                        else "-")
          + "; the model's time per episode: " + (f"{np.mean(seconds):.0f} s" if seconds else "-"))


def print_answers(paths: Sequence[str]) -> None:
    """The plan arm's requests: refusals, errors, tokens, cost and time per episode, from the answers files."""
    print("\nThe frontier plan's requests (the answers files):")
    if not paths:
        print("- no answers file (--answers, or $RUNS_DIR/fanswers/*.jsonl)")
        return
    print("| Answers file | Episodes | Refusals | Errors | Input tokens per episode | Output tokens per episode | "
          "Cost, USD | Seconds per episode |\n|---|---|---|---|---|---|---|---|")
    every = []
    for path in [*paths, None]:
        rows = every if path is None else list(frontier.read_answers(path).values())
        if path is not None:
            every += rows
        elif len(paths) < 2:
            continue
        if not rows:
            print(f"| {os.path.basename(path)} | 0 | | | | | | |")
            continue
        refusals = sum(r.get("error") == "refusal" for r in rows)
        errors = sum(bool(r.get("error")) for r in rows) - refusals
        usage = [r.get("usage") or {} for r in rows]
        tokens_in = np.mean([sum(u.get(f) or 0 for f in frontier_answers.USAGE_FIELDS if f != "output_tokens")
                             for u in usage])
        tokens_out = np.mean([u.get("output_tokens") or 0 for u in usage])
        cost = frontier_answers.cost_usd(rows)
        seconds = [r["seconds"] for r in rows if r.get("seconds") is not None]
        print(f"| {'all' if path is None else os.path.basename(path)} | {len(rows)} | {refusals} | {errors} | "
              f"{tokens_in:,.0f} | {tokens_out:,.0f} | {'-' if cost is None else f'{cost:.2f}'} | "
              + (f"{np.mean(seconds):.1f}" if seconds else "-") + " |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t1", default="iid16", help="T1's layout set (secondary)")
    ap.add_argument("--t3", default="iid17", help="T3's layout set")
    ap.add_argument("--t5", default="iid18", help="T5's layout set")
    ap.add_argument("--train", default="iid11", help="the layout set that the married system's ASM is trained on")
    ap.add_argument("--answers", nargs="*", default=None, help="the plan arm's answers files")
    a = ap.parse_args()
    fr = Frontier(a.t1, a.t3, a.t5, a.train)
    answers = sorted(glob.glob(os.path.join(runs.path("fanswers"), "*.jsonl"))) if a.answers is None else a.answers
    print(f"# Frontier-model study: T2 (the reworded sets on paper), T3 (the harder scenes on {a.t3}) and T5 (the new "
          f"scenes on {a.t5}), clean depth; T1 ({a.t1}) is secondary\n")
    for label, parts in (("Evals", TEST_PARTS), ("T1's evals (secondary)", ("T1",))):
        missing = []
        for row in ROWS:
            missing += [n for n in fr.row(row, parts)[1] if n not in missing]
        if missing:
            print(f"{label} that are not there: {', '.join(missing)}\n")
    print_success(fr)
    print_tests(fr)
    print("\n## Secondary (reported, no claims)")
    print_t1(fr)
    print_plans(fr)
    print_agent(fr)
    print_answers(answers)


if __name__ == "__main__":
    main()
