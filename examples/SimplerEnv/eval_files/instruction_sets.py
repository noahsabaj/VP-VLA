"""Reworded instructions for the 4 WidowX tasks: the same scene and the same goal, in other words.

The planner reads an instruction with one fixed rule (sam3_client.extract_target_from_task and
planner/planner_client.place_target_from_task): it drops a leading verb and "the"; the words before "on", "onto",
"in" or "into" name the object to move, and the words after it name the target. Then it runs the same
pick-and-place script. Each set below moves further from that form:

* synonyms: the same form in other words (a verb the rule does not know, another name for the target);
* attributes: the same form, but the objects are named by their kind or color;
* order: the same names, but not in the form "verb object on target";
* reasoning: the object to move must be worked out from what it is used for or known for.

`--instruction-set default` (the default) keeps SimplerEnv's own instruction.

The marriage study's controls (experiments README, "Marriage study") ask whether a VLA reads the instruction at all:
`blank` gives an empty instruction and `wrong` another task's own. They are not rewordings, so they stay out of
INSTRUCTION_SETS, which the studies of the reworded sets go through.
"""

from typing import Optional

TASKS = ("PutSpoonOnTableClothInScene-v0", "PutCarrotOnPlateInScene-v0",
         "StackGreenCubeOnYellowCubeBakedTexInScene-v0", "PutEggplantInBasketScene-v0")
# SimplerEnv's own instruction of each task (its get_language_instruction), in the order of TASKS.
OWN = ("put the spoon on the towel", "put carrot on plate", "stack the green block on the yellow block",
       "put eggplant into yellow basket")

# One instruction per task, in the order of TASKS. Fixed before any eval with them ran.
INSTRUCTION_SETS = {
    "synonyms": (
        "place the spoon on the cloth",
        "set the carrot on the dish",
        "put the green cube on top of the yellow cube",
        "drop the eggplant into the basket",
    ),
    "attributes": (
        "put the utensil on the towel",
        "put the orange vegetable on the plate",
        "stack the green object on the yellow object",
        "put the purple vegetable into the basket",
    ),
    "order": (
        "the towel is where the spoon should go",
        "the plate should get the carrot",
        "the yellow block should end up under the green block",
        "the basket is where the eggplant goes",
    ),
    "reasoning": (
        "put what you would eat soup with on the towel",
        "put the food that rabbits love on the plate",
        "put the block that is the color of grass on the other block",
        "put the vegetable that baba ganoush is made from into the basket",
    ),
}
# The controls, in the order of TASKS: the spoon and carrot tasks swap their own instructions, and so do the stack and
# eggplant tasks. Fixed on 2026-10-31, before any eval with them ran.
CONTROL_SETS = {
    "blank": ("",) * len(TASKS),
    "wrong": (OWN[1], OWN[0], OWN[3], OWN[2]),
}
NAMES = ("default", *INSTRUCTION_SETS, *CONTROL_SETS)


def instruction_for(set_name: str, env_name: str) -> Optional[str]:
    """The instruction of `set_name` for the task `env_name`; None for "default" (SimplerEnv's own)."""
    if set_name == "default":
        return None
    sets = {**INSTRUCTION_SETS, **CONTROL_SETS}
    if set_name not in sets:
        raise ValueError(f"unknown instruction set {set_name!r}; known: {', '.join(NAMES)}")
    if env_name not in TASKS:
        raise ValueError(f"the instruction sets cover only the 4 WidowX tasks, not {env_name}")
    return sets[set_name][TASKS.index(env_name)]
