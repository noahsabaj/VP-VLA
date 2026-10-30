"""Training rewordings of the instructions of the planner's demonstrations, and the exclusion list that keeps the test
sets' phrasings and concepts out of them.

REWORDINGS gives each of the 4 WidowX tasks 8 rewordings in each of the test sets' four families: other words
(synonyms), the objects named by what they look like (attributes), other sentence forms (order) and the objects named
by what they are used for or known for (reasoning). planner_demos_to_lerobot.py --reword pairs each demonstration with
one rewording of each family (rewording). The visual prompts keep the objects of SimplerEnv's own instruction.

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
import zlib
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


FAMILIES = ("synonyms", "attributes", "order", "reasoning")
# Written on 2026-10-30, after the exclusion list; the tests check each one against it.
REWORDINGS = {
    "PutSpoonOnTableClothInScene-v0": {
        "synonyms": (
            "lay the spoon on the napkin",
            "rest the teaspoon on the towel",
            "transfer the spoon onto the rag",
            "position the spoon on the tea towel",
            "shift the spoon onto the napkin",
            "deposit the spoon on the hand towel",
            "relocate the teaspoon to the towel",
            "carry the spoon over to the rag",
        ),
        "attributes": (
            "put the shiny metal one with a rounded end on the towel",
            "put the long silver one on the towel",
            "put the slim steel one on the towel",
            "put the metallic one on the towel",
            "lay the polished metal one on the towel",
            "put the thin gray one with a handle on the towel",
            "move the narrow silver one onto the towel",
            "put the small curved metal one on the towel",
        ),
        "order": (
            "onto the towel, put the spoon",
            "the spoon needs to be on the towel",
            "make sure the spoon is lying on the towel",
            "pick up the spoon, then lay it on the towel",
            "find the towel and put the spoon on it",
            "the spoon belongs on the towel",
            "I want the spoon on the towel",
            "take the spoon and lay it down on the towel",
        ),
        "reasoning": (
            "put what you stir tea with on the towel",
            "put the one you would use to scoop ice cream on the towel",
            "put what you measure sugar with on the towel",
            "put the one used to stir coffee on the towel",
            "put what you would stir a cup of cocoa with on the towel",
            "put the one a barista uses to stir a latte on the towel",
            "put the one you would use to mix honey into tea on the towel",
            "put what you would scoop sugar with on the towel",
        ),
    },
    "PutCarrotOnPlateInScene-v0": {
        "synonyms": (
            "lay the carrot on the platter",
            "rest the carrot on the saucer",
            "transfer the carrot to the platter",
            "position the carrot on the plate",
            "deposit the carrot onto the platter",
            "bring the carrot over to the saucer",
            "shift the carrot onto the serving platter",
            "lower the carrot onto the plate",
        ),
        "attributes": (
            "put the long tapered root on the plate",
            "put the pointy root on the plate",
            "put the slender root on the plate",
            "put the crunchy root on the round plate",
            "put the tapered one on the round plate",
            "put the long pointed one onto the plate",
            "put the narrow root onto the plate",
            "lay the cone-shaped root on the plate",
        ),
        "order": (
            "onto the plate, put the carrot",
            "the carrot needs to be on the plate",
            "make sure the carrot is resting on the plate",
            "pick up the carrot, then lay it on the plate",
            "find the plate and put the carrot on it",
            "the carrot belongs on the plate",
            "I want the carrot on the plate",
            "take the carrot and rest it on the plate",
        ),
        "reasoning": (
            "put what a snowman uses as a nose on the plate",
            "put the root said to help you see in the dark on the plate",
            "put the root rich in beta-carotene on the plate",
            "put what is often grated into coleslaw on the plate",
            "put the root that is often juiced for breakfast on the plate",
            "put the root used in a famous cake with cream cheese frosting on the plate",
            "put the root that is often cut into sticks for a lunchbox on the plate",
            "put the one a cook would peel and dice for a stir-fry on the plate",
        ),
    },
    "StackGreenCubeOnYellowCubeBakedTexInScene-v0": {
        "synonyms": (
            "pile the green block onto the yellow block",
            "rest the green brick on the yellow brick",
            "balance the green block upon the yellow block",
            "position the green brick onto the yellow brick",
            "heap the green brick onto the yellow brick",
            "perch the green block on the yellow block",
            "mount the green brick onto the yellow brick",
            "lay the green block on the yellow block",
        ),
        "attributes": (
            "stack the green one on the yellow one",
            "stack the small green block on the small yellow block",
            "stack the green square on the yellow square",
            "stack the green toy on the yellow toy",
            "put the green-colored block on the yellow-colored block",
            "stack the lime-colored block on the lemon-colored block",
            "stack the little green block on the little yellow block",
            "put the bright green block on the bright yellow block",
        ),
        "order": (
            "onto the yellow block, stack the green block",
            "the green block needs to be on the yellow block",
            "make sure the green block sits on the yellow block",
            "pick up the green block, then stack it on the yellow block",
            "find the yellow block and stack the green block on it",
            "the green block belongs on the yellow block",
            "I want the green block stacked on the yellow block",
            "first grab the green block, then rest it on the yellow block",
        ),
        "reasoning": (
            "stack the block that matches a lime on the block that matches a lemon",
            "stack the block whose color comes from mixing blue and yellow on the yellow block",
            "stack the block with the same hue as a frog on the block with the same hue as a banana",
            "stack the block the shade of a pine tree on the block the shade of a sunflower",
            "stack the block that shares its color with a cucumber on the yellow block",
            "stack the block colored like a leaf on the block colored like the sun",
            "stack the block that matches a pea on the block that matches a chick",
            "stack the block with the hue of an emerald on the block with the hue of a canary",
        ),
    },
    "PutEggplantInBasketScene-v0": {
        "synonyms": (
            "put the aubergine into the yellow basket",
            "toss the eggplant into the yellow bin",
            "lay the aubergine in the yellow basket",
            "deposit the eggplant in the yellow container",
            "lower the aubergine into the yellow bin",
            "pop the eggplant into the yellow crate",
            "stow the eggplant in the yellow basket",
            "tuck the aubergine into the yellow hamper",
        ),
        "attributes": (
            "put the dark glossy one into the yellow basket",
            "put the large oval one into the yellow basket",
            "put the shiny dark-skinned one into the yellow basket",
            "put the plump oval one into the yellow basket",
            "put the dark oblong one into the yellow basket",
            "put the heavy dark one into the basket",
            "put the smooth dark-skinned one in the yellow basket",
            "put the bulbous dark one into the yellow basket",
        ),
        "order": (
            "into the yellow basket, put the eggplant",
            "the eggplant needs to be in the yellow basket",
            "make sure the eggplant is in the yellow basket",
            "pick up the eggplant, then put it into the yellow basket",
            "find the yellow basket and put the eggplant in it",
            "the eggplant belongs in the yellow basket",
            "I want the eggplant in the yellow basket",
            "grab the eggplant and lower it into the yellow basket",
        ),
        "reasoning": (
            "put the main ingredient of moussaka into the yellow basket",
            "put what is sliced and fried for parmigiana into the yellow basket",
            "put the one that the British call an aubergine into the yellow basket",
            "put the nightshade with glossy dark skin into the yellow basket",
            "put what ratatouille and moussaka have in common into the yellow basket",
            "put the one known as brinjal in India into the yellow basket",
            "put the one that is the star of imam bayildi into the yellow basket",
            "put what is layered with cheese and tomato in parmigiana into the yellow basket",
        ),
    },
}


def rewording(env_name: str, family: str, episode_key: str) -> str:
    """The training rewording of `family` for an episode of the task `env_name`: one of the task's, always the same
    for the same episode (chosen by a checksum of the episode key)."""
    options = REWORDINGS[env_name][family]
    return options[zlib.crc32(f"{episode_key}|{family}".encode()) % len(options)]
