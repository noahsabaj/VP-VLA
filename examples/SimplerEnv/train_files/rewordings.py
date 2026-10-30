"""Training rewordings of the instructions of the planner's demonstrations, and the exclusion list that keeps the test
sets' phrasings and concepts out of them.

The exclusion list (TEST_INSTRUCTIONS and EXCLUDED) was fixed on 2026-10-30, before any training rewording was
written. A training rewording may not be a test instruction (compared in lower case, without punctuation) and may not
contain an excluded word or phrase (whole words). The test sets are:
- T2, the four reworded sets of the language study (eval_files/instruction_sets.py): synonyms, attributes, other word
  orders and reasoning;
- T3, the harder scenes (eval_files/harder_scenes.py): a plate named by where it is, and two pick-and-places in one
  instruction.
EXCLUDED holds, for each test family, the words that carry its concepts (its new names, verbs, structures and facts)
with their close variants. SimplerEnv's own names of the objects stay allowed, as the training keeps its own
instructions too.
"""

import re
from typing import List

from examples.SimplerEnv.eval_files.harder_scenes import TASKS as HARDER_TASKS
from examples.SimplerEnv.eval_files.instruction_sets import INSTRUCTION_SETS, TASKS

# SimplerEnv's own instruction of each of the 4 WidowX tasks (their get_language_instruction).
OWN = dict(zip(TASKS, ("put the spoon on the towel", "put carrot on plate", "stack the green block on the yellow block",
                       "put eggplant into yellow basket"), strict=True))
TEST_INSTRUCTIONS = tuple(i for s in INSTRUCTION_SETS.values() for i in s) + tuple(
    t.instruction for t in HARDER_TASKS.values() if t.instruction not in OWN.values())
# Regular expressions, each matched as whole words in the lower-case instruction without punctuation.
EXCLUDED = {
    # T2 synonyms: "place the spoon on the cloth", "set the carrot on the dish", "put the green cube on top of the
    # yellow cube", "drop the eggplant into the basket": their verbs, their new names and "on top of".
    "synonyms": (r"plac(e|es|ed|ing)", r"set(s|ting)?", r"drop(s|ped|ping)?", r"\w*cloths?\w*", r"\w*dish\w*",
                 r"cub(e|es|ic)", r"on top of", r"atop"),
    # T2 attributes: "the utensil", "the orange vegetable", "the green object ... the yellow object", "the purple
    # vegetable": the kind words and their synonyms, and the colours that stand for an object that SimplerEnv does not
    # name by its colour.
    "attributes": (r"utensils?", r"cutlery", r"silverware", r"flatware", r"tools?", r"vegetables?", r"veg(gie|gies)?",
                   r"foods?", r"objects?", r"things?", r"items?", r"pieces?", r"orange", r"purple", r"violet"),
    # T2 order: "the towel is where the spoon should go", "the plate should get the carrot", "the yellow block should
    # end up under the green block", "the basket is where the eggplant goes": the words of these structures.
    "order": (r"should", r"where", r"end(s|ed)? up", r"under(neath)?", r"below", r"beneath", r"go(es|ing|ne)?",
              r"get(s|ting)?", r"receives?"),
    # T2 reasoning: "what you would eat soup with", "the food that rabbits love", "the block that is the color of
    # grass on the other block", "the vegetable that baba ganoush is made from": the facts and these forms.
    "reasoning": (r"eat(s|ing|en)?", r"soups?", r"broth", r"stews?", r"rabbits?", r"bunn(y|ies)", r"hares?",
                  r"lov(e|es|ed)", r"grass(y|es)?", r"lawns?", r"turf", r"colou?r of", r"(an)?other", r"baba",
                  r"gh?anou(sh|j)", r"mutabb?al", r"made (from|of)"),
    # T3: "the plate that is closer to (farther from) the camera".
    "harder scenes": (r"camera", r"clos(e|er|est)", r"near(er|est|by)?", r"far(ther|thest)?", r"furthe(r|st)"),
}


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def exclusions(instruction: str) -> List[str]:
    """Why `instruction` may not be a training rewording: the test instruction that it is, and the excluded words that
    it holds ("FAMILY: WORDS"). Empty: it may."""
    text = normalize(instruction)
    found = [f"test instruction: {t}" for t in TEST_INSTRUCTIONS if normalize(t) == text]
    for family, patterns in EXCLUDED.items():
        for pattern in patterns:
            match = re.search(rf"\b(?:{pattern})\b", text)
            if match:
                found.append(f"{family}: {match.group(0)}")
    return found
