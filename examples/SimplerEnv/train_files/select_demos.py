"""The marriage study's training sets (the experiments README, "Marriage study": "Demonstrations" and the change of
2026-10-30): which recorded episodes each model trains on, and the sets themselves.

    python -m examples.SimplerEnv.train_files.select_demos --runs RUNS_DIR
    python -m examples.SimplerEnv.train_files.select_demos --runs RUNS_DIR --write SET --data-root DATA_ROOT \
        --stats CHECKPOINT_RUN/dataset_statistics.json --visual-prompts VP_DIR

The episodes come from five pools (POOLS), each a run folder in RUNS_DIR recorded with RECORD_DEMOS=1 and each with
the plan's rule for how far it grows. The first command prints, for each pool and task, the recorded episodes, the
usable ones (the successes; of the corrections, the episodes in which the planner took over and that succeeded) and
what the selection needs, with the chunks still to run in the runner's arguments: a task with fewer than it needs gets
its pool's next chunk, up to the pool's last episode. Then it says which sets are ready. The second also writes one
set (SETS) into DATA_ROOT/SET with planner_demos_to_lerobot.py's write_dataset (the reworded sets with --reword), and
its visual prompts into VP_DIR/SET. A set is written only when every pool that it takes from is done; its
meta/sources.jsonl lists its recorded episodes.

The sets, with N = 250 of each of the 9 training tasks:
- planner_demos (VLA-P): the first N successes of each task, in episode order: of SimplerEnv's 4 tasks the planner's
  with its fixed rule, of the 5 harder scenes the planner's reading the VLM's marks.
- planner_demos_reworded (VLA-PL): the same episodes, with the training rewordings.
- targeted_demos_reworded (VLA-PLT): of each of SimplerEnv's 4 tasks, the TOP successes with the highest score
  P(planner succeeds) x (1 - P(VLA succeeds)) of the ASM (--asm; the lower episode on a tie) and N - TOP drawn at
  random from the task's other successes (each task its own draw, seed 0); of the harder scenes, VLA-PL's episodes.
- targeted_corrections_reworded (VLA-PLTD): VLA-PLT's episodes and the first N successful corrections of each of
  SimplerEnv's 4 tasks, in one set. A mixture of two sets would draw from each half of the time (the training's
  get_vla_dataset_with_inline_vp does not weight the sets by their size); one set gives every step the same chance.
- vla_successes_reworded (VLA-S): VLA-0's first successes of each task, in episode order, as many as VLA-PL has of the
  task. A harder scene with fewer after its pool's last episode leaves a gap, which VLA-0's next successes of
  SimplerEnv's 4 tasks fill in equal shares (the first tasks in order take the remainder), so that VLA-S has as many
  as VLA-PL.
- planner_demos_450 and planner_demos_1125 (the learning curve): VLA-P's first 50 and 125 of each task.
A task that has fewer than it needs after its pool's last episode keeps what it has, and the report says so.
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from examples.SimplerEnv.eval_files import harder_scenes as hs
from examples.SimplerEnv.eval_files.planner import selector as s
from examples.SimplerEnv.train_files.planner_demos_to_lerobot import load_meta, summary_lines, write_dataset
from examples.SimplerEnv.train_files.rewordings import FAMILIES

N = 250  # demonstrations of each training task
TOP = 188  # VLA-PLT: three quarters of N (rounded up) by score, the others drawn at random
SEED = 0  # VLA-PLT's random draw
TOOK_OVER = ("holding", "grasp")  # a correction: the planner took over (planner_client.py, "VLA then planner")
TASKS, HARDER = tuple(s.TASKS), tuple(hs.TASKS)


class Pool(NamedTuple):
    """The recorded episodes of one run folder, and how far the plan lets it grow."""

    run: str  # the run folder in RUNS_DIR
    what: str  # what drove the episodes
    tasks: Tuple[str, ...]
    layout_seed: int  # iidN: a random layout of its own for each episode, from layout seed N
    first: int  # the first run's episodes of each task: 0 to first - 1
    step: int  # then this many more at a time for a task that needs more
    last: Optional[int]  # at most this many episodes of a task (None: no limit)
    job: str  # a chunk in the runner's arguments
    chunk: Optional[int] = None  # episodes in each printed chunk (None: one chunk for each task)
    corrections: bool = False  # usable: the planner took over and the episode succeeded (otherwise: a success)


POOLS = {
    "planner": Pool("demos_planner_iid100", "the planner, fixed rule", TASKS, 100, 700, 0, 700,
                    "{task}:{start}:{end}"),
    "planner_harder": Pool("demos_vp_harder_iid100", "the planner reading the VLM's marks", HARDER, 100, 700, 300, 2000,
                           "{task}:{start}:{end}"),
    "vla": Pool("demos_vla_iid100", "VLA-0", TASKS, 100, 400, 100, None, "{task}:{start}:{end}", chunk=8),
    "vla_harder": Pool("demos_vla_harder_iid100", "VLA-0", HARDER, 100, 400, 100, 600, "{task}:{start}:{end}", chunk=8),
    "corrections": Pool("demos_corrections_iid101", "VLA-PLT, then the planner", TASKS, 101, 100, 100, None,
                        "none:{task}:{start}:{end}", corrections=True),
}


class TrainingSet(NamedTuple):
    model: str
    pools: Tuple[str, ...]  # the pools it takes from
    reword: bool  # also each episode with a training rewording of each family
    asm: bool = False  # needs the ASM


PLANNER_POOLS = ("planner", "planner_harder")
SETS = {
    "planner_demos": TrainingSet("VLA-P", PLANNER_POOLS, False),
    "planner_demos_reworded": TrainingSet("VLA-PL", PLANNER_POOLS, True),
    "targeted_demos_reworded": TrainingSet("VLA-PLT", PLANNER_POOLS, True, asm=True),
    "targeted_corrections_reworded": TrainingSet("VLA-PLTD", (*PLANNER_POOLS, "corrections"), True, asm=True),
    "vla_successes_reworded": TrainingSet("VLA-S", (*PLANNER_POOLS, "vla_harder", "vla"), True),
    "planner_demos_450": TrainingSet("VLA-P, learning curve", PLANNER_POOLS, False),
    "planner_demos_1125": TrainingSet("VLA-P, learning curve", PLANNER_POOLS, False),
}
CURVE = {"planner_demos_450": 50, "planner_demos_1125": 125}  # VLA-P's first demonstrations of each task


class Episodes(NamedTuple):
    """A task's recorded episodes in a pool."""

    recorded: int  # episodes 0 to recorded - 1
    usable: List[Tuple[int, str]]  # (episode, folder) of each usable one before the pool's last episode, in order


class Selection(NamedTuple):
    sets: Dict[str, Optional[List[str]]]  # each set's recorded episodes (folders); None: not ready
    waiting: Dict[str, str]  # what each set that is not ready waits for
    report: List[str]


def episode_list(episodes: List[int]) -> str:
    """Episode numbers, with runs of them as ranges: "3, 7-9"."""
    parts, start = [], None
    for i, e in enumerate(episodes):
        start = e if start is None else start
        if i + 1 == len(episodes) or episodes[i + 1] != e + 1:
            parts.append(f"{start}" if start == e else f"{start}-{e}")
            start = None
    return ", ".join(parts)


def is_usable(meta: dict, pool: Pool) -> bool:
    ok = bool(meta["success"]) and meta["steps"] > 0
    return (ok and meta.get("takeover") in TOOK_OVER) if pool.corrections else ok


def read_pool(runs: str, pool: Pool) -> Optional[Dict[str, Episodes]]:
    """Each task's recorded episodes in the pool's run folder (folders with meta.json and episode.npz at any depth;
    None when the run folder does not exist). A gap in a task's episodes, a second recording of an episode, or an
    episode of another task or of other layouts stops the load."""
    root = os.path.join(runs, pool.run)
    if not os.path.isdir(root):
        return None
    found: Dict[str, Dict[int, Tuple[str, dict]]] = defaultdict(dict)
    for dirpath, _, files in os.walk(root):
        if "meta.json" not in files or "episode.npz" not in files:
            continue
        folder, meta = os.path.normpath(dirpath), load_meta(dirpath)
        task, episode = meta["env_name"], int(meta["obj_episode_id"])
        if task not in pool.tasks:
            raise SystemExit(f"{folder}: {task} is not one of the tasks of {pool.run}")
        if meta.get("layout_seed") != pool.layout_seed or not meta.get("layout_per_episode"):
            raise SystemExit(f"{folder}: layout seed {meta.get('layout_seed')}, a layout per episode "
                             f"{meta.get('layout_per_episode')}; {pool.run} has iid{pool.layout_seed}")
        if episode in found[task]:
            raise SystemExit(f"{folder}: {task} episode {episode} is also in {found[task][episode][0]}")
        found[task][episode] = (folder, meta)
    out = {}
    for task in pool.tasks:
        episodes = found.get(task, {})
        recorded = max(episodes, default=-1) + 1
        missing = sorted(set(range(recorded)) - set(episodes))
        if missing:
            raise SystemExit(f"{root}: no recording of {task} episode {episode_list(missing)}")
        end = recorded if pool.last is None else min(recorded, pool.last)
        out[task] = Episodes(recorded, [(e, episodes[e][0]) for e in range(end) if is_usable(episodes[e][1], pool)])
    return out


def is_done(pool: Pool, episodes: Episodes, need: int) -> bool:
    """The task's first run is complete, and it has what it needs or the pool's last episode."""
    return episodes.recorded >= pool.first and (
        len(episodes.usable) >= need or (pool.last is not None and episodes.recorded >= pool.last))


def next_chunks(pool: Pool, task: str, episodes: Episodes, need: int) -> List[str]:
    """The task's next episodes as the runner's chunks: none when it is done or its first run is not complete."""
    if not pool.step or episodes.recorded < pool.first or is_done(pool, episodes, need):
        return []
    start = episodes.recorded
    end = start + pool.step if pool.last is None else min(start + pool.step, pool.last)
    size = pool.chunk or end - start
    return [pool.job.format(task=task, start=a, end=min(a + size, end)) for a in range(start, end, size)]


def pool_report(pool: Pool, eps: Optional[Dict[str, Episodes]], needs: Dict[str, int], note: str = "") -> List[str]:
    """The report's lines of a pool: each task's episodes, what it needs and its next chunks."""
    grow = f", then {pool.step} more at a time" if pool.step else ""
    limit = f", up to {pool.last}" if pool.last is not None and pool.last > pool.first else ""
    head = f"{pool.run} ({pool.what}; iid{pool.layout_seed}; episodes 0 to {pool.first - 1}{grow}{limit})"
    if eps is None:
        return [f"{head}: not run yet"]
    lines = [f"{head}:"]
    kind = "successful corrections" if pool.corrections else "successes"
    for task in pool.tasks:
        e, need = eps[task], needs[task]
        seen = e.recorded if pool.last is None else min(e.recorded, pool.last)
        rate = f" ({100 * len(e.usable) / seen:.1f}%)" if seen else ""
        unused = f" (episodes from {pool.last} on are not used)" if seen < e.recorded else ""
        chunks = next_chunks(pool, task, e, need)
        if e.recorded < pool.first:
            status = f"its first run has {e.recorded} of {pool.first} episodes"
        elif chunks:
            status = "needs more: " + " ".join(chunks)
        elif len(e.usable) < need:
            status = f"done, short: keeps {len(e.usable)}"
        else:
            status = "done"
        lines.append(f"  {task}: {e.recorded} episodes{unused}, {len(e.usable)} {kind}{rate}; needs {need}{note}: "
                     f"{status}")
    return lines


def targeted(runs: str, pool: Pool, eps: Dict[str, Episodes], asm: s.TwoModelASM, n: int = N, top: int = TOP,
             seed: int = SEED) -> Tuple[Dict[str, List[Tuple[int, str]]], List[str]]:
    """VLA-PLT's episodes of each task of the pool, chosen from each successful episode's first look (its planner
    record, planner_stats.jsonl) by the ASM's score, and the report's lines: the mean score of the best-scored, of the
    drawn and of all of each task's successes."""
    records = s.planner_records(os.path.join(runs, pool.run), list(pool.tasks))
    depth = s.planner_settings(records).get("depth_noise")
    trained = (asm.trained_on or {}).get("depth_noise")
    if trained is not None and trained != depth:
        raise SystemExit(f"the ASM is trained on {trained} depth, {pool.run} has {depth} depth")
    by_key = {(r["task_folder"], r["obj_episode_id"]): r for r in records}
    chosen, lines = {}, []
    for task in pool.tasks:
        ok = eps[task].usable
        missing = [e for e, _ in ok if (task, e) not in by_key]
        if missing:
            raise SystemExit(f"{pool.run}: no planner record of {task} episode {episode_list(missing)}")
        failed = [e for e, _ in ok if not by_key[task, e]["success"]]
        if failed:
            raise SystemExit(f"{pool.run}: {task} episode {episode_list(failed)} succeeded in the recording but not in "
                             "the planner record")
        if len(ok) <= n:
            chosen[task] = ok
            lines.append(f"  {task}: all {len(ok)} successes")
            continue
        p_plan, p_vla = asm.probs([by_key[task, e] for e, _ in ok], [task] * len(ok))
        score = p_plan * (1 - p_vla)
        order = sorted(range(len(ok)), key=lambda i: (-score[i], ok[i][0]))
        best, rest = order[:top], sorted(order[top:])
        drawn = np.random.default_rng(seed).choice(rest, n - top, replace=False).tolist()
        chosen[task] = [ok[i] for i in sorted(best + drawn)]
        lines.append(f"  {task}: mean score {score[best].mean():.3f} of the best {top}, {score[drawn].mean():.3f} of "
                     f"the {n - top} drawn, {score.mean():.3f} of all {len(ok)} successes")
    return chosen, lines


def folders(parts: Dict[str, List[Tuple[int, str]]]) -> List[str]:
    return [folder for episodes in parts.values() for _, folder in episodes]


def select(runs: str, asm_path: Optional[str], pools: Dict[str, Pool] = POOLS, n: int = N, top: int = TOP,
           curve: Dict[str, int] = CURVE) -> Selection:
    """Each set's recorded episodes (None: not ready yet) and the report (see the module docstring)."""
    eps = {name: read_pool(runs, pool) for name, pool in pools.items()}
    planner_tasks = [t for name in PLANNER_POOLS for t in pools[name].tasks]
    base, harder = pools["vla"].tasks, pools["vla_harder"].tasks
    needs = {name: dict.fromkeys(pools[name].tasks, n) for name in pools}

    def done(name: str) -> bool:
        return eps[name] is not None and all(is_done(pools[name], eps[name][t], needs[name][t])
                                             for t in pools[name].tasks)

    # VLA-P: the first n successes of each task. VLA-S: as many of each task, the harder scenes' gap filled from
    # SimplerEnv's 4 tasks (before the planner's pools are done, n of each).
    planner_ready = all(done(name) for name in PLANNER_POOLS)
    demos = {t: eps[name][t].usable[:n] for name in PLANNER_POOLS for t in pools[name].tasks} if planner_ready else {}
    needs["vla_harder"] = {t: len(demos[t]) if planner_ready else n for t in harder}
    note = " (and a share of the harder scenes' gap, when it is known)"
    fill = dict.fromkeys(base, 0)
    if planner_ready and done("vla_harder"):
        gap = sum(k - min(k, len(eps["vla_harder"][t].usable)) for t, k in needs["vla_harder"].items())
        share, extra = divmod(gap, len(base))
        fill, note = {t: share + int(i < extra) for i, t in enumerate(base)}, ""
    needs["vla"] = {t: (len(demos[t]) if planner_ready else n) + fill[t] for t in base}

    report = []
    for name, pool in pools.items():
        report += pool_report(pool, eps[name], needs[name], note if name == "vla" else "")
    pool_done = {name: done(name) for name in pools}
    asm = None
    if asm_path is not None and os.path.exists(asm_path):
        asm = s.load_asm(asm_path)
        if not isinstance(asm, s.TwoModelASM):
            raise SystemExit(f"{asm_path}: not a two-model ASM (VLA-PLT's score needs P(VLA succeeds))")

    sets: Dict[str, Optional[List[str]]] = dict.fromkeys(SETS)
    waiting = {}
    for name, spec in SETS.items():
        why = [pools[p].run for p in spec.pools if not pool_done[p]]
        if spec.asm and asm is None:
            why.append(f"the ASM ({asm_path}: no such file)")
        if why:
            waiting[name] = "waiting for " + ", ".join(why)
    if planner_ready:
        sets["planner_demos"] = sets["planner_demos_reworded"] = folders(demos)
        for name, k in curve.items():
            sets[name] = folders({t: v[:k] for t, v in demos.items()})
        if asm is not None:
            chosen, lines = targeted(runs, pools["planner"], eps["planner"], asm, n, top)
            report += [f"VLA-PLT's choice ({asm_path}):", *lines]
            chosen.update({t: demos[t] for t in pools["planner_harder"].tasks})  # the harder scenes: VLA-PL's
            sets["targeted_demos_reworded"] = folders(chosen)
            if pool_done["corrections"]:
                corrections = {t: eps["corrections"][t].usable[:n] for t in pools["corrections"].tasks}
                sets["targeted_corrections_reworded"] = sets["targeted_demos_reworded"] + folders(corrections)
        if pool_done["vla_harder"] and pool_done["vla"]:
            vla = {**{t: eps["vla"][t].usable[:k] for t, k in needs["vla"].items()},
                   **{t: eps["vla_harder"][t].usable[:k] for t, k in needs["vla_harder"].items()}}
            sets["vla_successes_reworded"] = folders({t: vla[t] for t in planner_tasks})
            if len(sets["vla_successes_reworded"]) != len(sets["planner_demos"]):
                raise SystemExit(f"VLA-S has {len(sets['vla_successes_reworded'])} episodes and VLA-PL "
                                 f"{len(sets['planner_demos'])}")
    report.append("sets:")
    for name, spec in SETS.items():
        got = sets[name]
        copies = 1 + len(FAMILIES) if spec.reword else 1
        report.append(f"  {name} ({spec.model}): " + (waiting[name] if got is None else
                      f"ready, {len(got)} recorded episodes, {copies * len(got)} in the set"))
    return Selection(sets, waiting, report)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="the folder of the run folders (RUNS_DIR)")
    ap.add_argument("--asm", help="VLA-PLT's ASM (default: RUNS_DIR/energy_model/asm_none.json)")
    ap.add_argument("--write", choices=list(SETS), help="also write this set")
    ap.add_argument("--data-root", help="with --write: the folder of the sets (DATA_ROOT)")
    ap.add_argument("--stats", help='with --write: the checkpoint run\'s dataset_statistics.json, or "data"')
    ap.add_argument("--visual-prompts", metavar="VP_DIR", help="with --write: the folder of the visual prompts")
    a = ap.parse_args()
    if a.write and not (a.data_root and a.stats and a.visual_prompts):
        ap.error("--write needs --data-root, --stats and --visual-prompts")
    selection = select(a.runs, a.asm or os.path.join(a.runs, "energy_model", "asm_none.json"))
    print("\n".join(selection.report))
    if a.write:
        chosen = selection.sets[a.write]
        if chosen is None:
            raise SystemExit(f"{a.write} is not ready: {selection.waiting[a.write]}")
        out = os.path.join(a.data_root, a.write)
        summary = write_dataset(out, chosen, a.stats, a.visual_prompts, SETS[a.write].reword)
        print("\n".join(summary_lines(out, summary)))


if __name__ == "__main__":
    main()
