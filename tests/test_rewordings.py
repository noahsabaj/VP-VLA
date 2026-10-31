"""CPU tests for the training rewordings and their exclusion list (train_files/rewordings.py)."""

from examples.SimplerEnv.eval_files.harder_scenes import TASKS as HARDER_TASKS
from examples.SimplerEnv.eval_files.instruction_sets import INSTRUCTION_SETS, TASKS
from examples.SimplerEnv.eval_files.novel_scenes import KINDS
from examples.SimplerEnv.eval_files.novel_scenes import TASKS as NOVEL_TASKS
from examples.SimplerEnv.train_files import rewordings as rw


def test_every_test_instruction_is_excluded():
    assert len(rw.TEST_INSTRUCTIONS) == 16 + 8  # T2's four sets and T5's instructions
    for instruction in rw.TEST_INSTRUCTIONS:
        assert rw.exclusions(instruction)[0] == f"test instruction: {instruction}"
    assert rw.exclusions("Put the eggplant on the plate!")[0].startswith("test instruction")


def test_each_test_family_is_caught_by_its_own_words():
    for family, instructions in INSTRUCTION_SETS.items():
        for instruction in instructions:
            assert any(f.startswith(f"{family}: ") for f in rw.exclusions(instruction)), instruction
    for name, kind in KINDS.items():  # T5's new objects and reasoning; its new pairs and long tasks have no own words
        if kind in ("new object", "reasoning"):
            assert any(f.startswith("new scenes: ") for f in rw.exclusions(NOVEL_TASKS[name].instruction)), name
    assert "new scenes: closer to the towel" in rw.exclusions(NOVEL_TASKS["NovelCarrotByTowelOnPlate-v0"].instruction)


def test_every_training_rewording_passes_the_exclusion_list():
    assert set(rw.REWORDINGS) == set(TASKS) | set(HARDER_TASKS)  # the harder scenes since the change of 2026-10-30
    own = {rw.normalize(i) for i in [*rw.OWN.values(), *(t.instruction for t in HARDER_TASKS.values())]}
    seen = set()
    for task, families in rw.REWORDINGS.items():
        assert tuple(families) == rw.FAMILIES
        for family, options in families.items():
            assert len(options) == 8, (task, family)
            for option in options:
                assert rw.exclusions(option) == [], (task, family, option)
                assert rw.normalize(option) not in own
                assert rw.normalize(option) not in seen, option
                seen.add(rw.normalize(option))


def test_each_episode_keeps_its_rewordings():
    task = TASKS[1]
    first = rw.rewording(task, "order", f"{task}/iid90/7")
    assert first in rw.REWORDINGS[task]["order"] and rw.rewording(task, "order", f"{task}/iid90/7") == first
    chosen = {rw.rewording(task, "reasoning", f"{task}/iid90/{i}") for i in range(200)}
    assert chosen == set(rw.REWORDINGS[task]["reasoning"])  # the episodes spread over all of them


def test_simplerenvs_own_instructions_are_allowed():
    assert rw.OWN[TASKS[0]] == HARDER_TASKS["ClutteredSpoonOnTowel-v0"].instruction
    assert rw.OWN[TASKS[1]] == HARDER_TASKS["ClutteredCarrotOnPlate-v0"].instruction
    for instruction in rw.OWN.values():
        assert rw.exclusions(instruction) == []
    assert rw.exclusions("Lay the spoon down on the napkin.") == []
    assert rw.exclusions("Put the spoon on the Table-Cloth") == ["synonyms: cloth"]
    # the harder scenes are in the training since the change of 2026-10-30: their words are allowed
    for instruction in (t.instruction for t in HARDER_TASKS.values()):
        assert rw.exclusions(instruction) == [], instruction
    assert rw.exclusions("put the spoon that is next to the towel on the plate") == ["new scenes: next to the towel"]
    assert rw.exclusions("don't put the can down") == ["new scenes: can", "new scenes: don t"]
