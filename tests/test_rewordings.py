"""CPU tests for the training rewordings and their exclusion list (train_files/rewordings.py)."""

from examples.SimplerEnv.eval_files.harder_scenes import TASKS as HARDER_TASKS
from examples.SimplerEnv.eval_files.instruction_sets import INSTRUCTION_SETS, TASKS
from examples.SimplerEnv.train_files import rewordings as rw


def test_every_test_instruction_is_excluded():
    assert len(rw.TEST_INSTRUCTIONS) == 16 + 3  # T2's four sets, and T3's instructions that are not SimplerEnv's own
    for instruction in rw.TEST_INSTRUCTIONS:
        assert rw.exclusions(instruction)[0] == f"test instruction: {instruction}"
    assert rw.exclusions("Put the carrot on the plate that is closer to the camera!")[0].startswith("test instruction")


def test_each_test_family_is_caught_by_its_own_words():
    for family, instructions in INSTRUCTION_SETS.items():
        for instruction in instructions:
            assert any(f.startswith(f"{family}: ") for f in rw.exclusions(instruction)), instruction
    for name in ("CarrotOnNearPlate-v0", "CarrotOnFarPlate-v0"):
        assert "harder scenes: camera" in rw.exclusions(HARDER_TASKS[name].instruction)


def test_simplerenvs_own_instructions_are_allowed():
    assert rw.OWN[TASKS[0]] == HARDER_TASKS["ClutteredSpoonOnTowel-v0"].instruction
    assert rw.OWN[TASKS[1]] == HARDER_TASKS["ClutteredCarrotOnPlate-v0"].instruction
    for instruction in rw.OWN.values():
        assert rw.exclusions(instruction) == []
    assert rw.exclusions("Lay the spoon down on the napkin.") == []
    assert rw.exclusions("Put the spoon on the Table-Cloth") == ["synonyms: cloth"]
    assert rw.exclusions("close the gripper around the carrot") == ["harder scenes: close"]
