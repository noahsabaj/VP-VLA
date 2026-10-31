"""Training rewordings of the instructions of the planner's demonstrations, and the exclusion list that keeps the test
sets' phrasings and concepts out of them.

REWORDINGS gives each of SimplerEnv's 4 WidowX tasks and each of the 5 harder scenes 8 rewordings in each of the test
sets' four families: other words (synonyms), the objects named by what they look like (attributes), other sentence
forms (order) and the objects named by what they are used for or known for (reasoning).
planner_demos_to_lerobot.py --reword pairs each demonstration with one rewording of each family (rewording). The visual
prompts keep the objects of the task's own instruction.

The exclusion list (TEST_INSTRUCTIONS and EXCLUDED) was fixed on 2026-10-30, before any training rewording was
written. A training rewording may not be a test instruction (compared in lower case, without punctuation) and may not
contain an excluded word or phrase (whole words). The test sets are:
- T2, the four reworded sets of the language study (eval_files/instruction_sets.py): synonyms, attributes, other word
  orders and reasoning;
- T5, the new test scenes (eval_files/novel_scenes.py): new objects, an object and a target that no training task
  pairs, an object named by what it is not or by the object next to it, and long tasks (a stack built on the plate,
  three pick-and-places in one instruction).
EXCLUDED holds, for each test family, the words that carry its concepts (its new names, verbs, structures and facts)
with their close variants. SimplerEnv's own names of the objects stay allowed, as the training keeps its own
instructions too.

Change of 2026-10-30, before any rewording of the harder scenes was written: the harder scenes (T3,
eval_files/harder_scenes.py) are now in the training (the marriage study's change of that date), so their
instructions and their family's words ("camera", "closer", "farther" ...) left the list, and T5 came in.
"""

import re
import zlib
from typing import List

from examples.SimplerEnv.eval_files.instruction_sets import INSTRUCTION_SETS, TASKS
from examples.SimplerEnv.eval_files.novel_scenes import TASKS as NOVEL_TASKS

# SimplerEnv's own instruction of each of the 4 WidowX tasks (their get_language_instruction).
OWN = dict(zip(TASKS, ("put the spoon on the towel", "put carrot on plate", "stack the green block on the yellow block",
                       "put eggplant into yellow basket"), strict=True))
TEST_INSTRUCTIONS = tuple(i for s in INSTRUCTION_SETS.values() for i in s) + tuple(
    t.instruction for t in NOVEL_TASKS.values())
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
    # T5: the new objects ("the sponge", "the soda can"), and an object named by what it is not ("the vegetable that
    # is not orange") or by the object next to it ("the carrot that is closer to the towel"). Its new pairs and long
    # tasks have no words of their own: their instructions are excluded as test instructions.
    "new scenes": (r"sponges?", r"soda", r"cans?", r"coke", r"cola", r"drinks?", r"beverages?", r"tins?", r"not",
                   r"\w+n t", r"except", r"excluding", r"rather than", r"instead of", r"apart from",
                   r"(next to|beside|by|near(er|est)?|clos(e|er|est) to) the (?!camera\b)\w+"),
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
    # The harder scenes (harder_scenes.py), written after the list's change of 2026-10-30. Their tables hold the
    # spoon, the towel, the carrot and the plate, or the carrot and two plates, so each rewording fits one of them
    # only, and those of the two-plate tasks keep the plate's place (nearer to or farther from the camera).
    "ClutteredSpoonOnTowel-v0": {
        "synonyms": (
            "move the spoon onto the towel",
            "lay the teaspoon on the hand towel",
            "rest the spoon upon the towel",
            "put the tablespoon on the towel",
            "bring the spoon to the towel",
            "lower the spoon onto the napkin",
            "transfer the teaspoon onto the towel",
            "lay the spoon down on the rag",
        ),
        "attributes": (
            "put the shiny one with a handle on the towel",
            "put the metal one on the soft square",
            "put the silver one with a small bowl at its end on the towel",
            "move the steel one onto the towel",
            "lay the gleaming metal one on the towel",
            "put the long thin metal one on the towel",
            "put the reflective one on the fabric square",
            "put the slender silver one onto the soft towel",
        ),
        "order": (
            "on the towel, lay the spoon",
            "the spoon must sit on the towel",
            "have the spoon resting on the towel",
            "the spoon is to be put on the towel",
            "leave the spoon lying on the towel",
            "grab the spoon and lay it on the towel",
            "it is the spoon that belongs on the towel",
            "the towel needs the spoon on it",
        ),
        "reasoning": (
            "put what you would stir a mug of tea with on the towel",
            "put the one used to scoop sugar into coffee on the towel",
            "put what you would use to scoop baking powder on the towel",
            "put the one you would dip into a jar of honey on the towel",
            "put what you would use to stir milk into coffee on the towel",
            "put the spoon on what you would dry your hands with",
            "put the spoon on what you would wipe a wet counter with",
            "put the one used to scoop flour into a bowl on the towel",
        ),
    },
    "ClutteredCarrotOnPlate-v0": {
        "synonyms": (
            "move the carrot onto the plate",
            "lay the carrot upon the platter",
            "rest the carrot on the serving plate",
            "bring the carrot to the platter",
            "put the carrot onto the saucer",
            "transfer the carrot onto the plate",
            "lower the carrot onto the saucer",
            "lay the carrot down on the plate",
        ),
        "attributes": (
            "put the tapered root on the plate",
            "put the pointed root on the round plate",
            "put the thin root onto the plate",
            "lay the cone-shaped one on the round plate",
            "put the skinny root on the plate",
            "put the root with a pointed tip on the flat round plate",
            "move the slim tapering root onto the plate",
            "put the bright root on the round plate",
        ),
        "order": (
            "on the plate, lay the carrot",
            "the carrot must sit on the plate",
            "have the carrot resting on the plate",
            "the carrot is to be put on the plate",
            "leave the carrot lying on the plate",
            "grab the carrot and lay it on the plate",
            "it is the carrot that belongs on the plate",
            "the plate needs the carrot on it",
        ),
        "reasoning": (
            "put the root that is often roasted with honey on the plate",
            "put the root you would cut into sticks for dipping in hummus on the plate",
            "put the root often shredded into a salad on the plate",
            "put the root pickled for a banh mi sandwich on the plate",
            "put the root that grows underground with feathery green leaves on the plate",
            "put the root that gajar halwa is cooked from on the plate",
            "put the root sold as baby sticks in snack bags on the plate",
            "put the carrot on what you would serve a slice of cake on",
        ),
    },
    # One part for each goal, the spoon's first (planner_demos_to_lerobot.py takes part N for goal N).
    "TwoStepSpoonAndCarrot-v0": {
        "synonyms": (
            "lay the spoon on the towel and the carrot on the platter",
            "move the spoon onto the napkin and the carrot onto the plate",
            "rest the teaspoon on the towel, then the carrot on the plate",
            "transfer the spoon to the towel and the carrot to the platter",
            "lower the spoon onto the rag and the carrot onto the saucer",
            "bring the spoon to the hand towel and the carrot to the plate",
            "lay the spoon upon the towel and the carrot upon the plate",
            "shift the spoon onto the towel, and the carrot onto the serving plate",
        ),
        "attributes": (
            "put the metal one on the towel and the tapered root on the plate",
            "put the shiny one on the soft square and the pointed root on the round plate",
            "put the long silver one on the towel and the thin root on the plate",
            "lay the steel one on the towel and the cone-shaped root on the plate",
            "put the gleaming one on the fabric square and the slender root on the plate",
            "put the one with a handle on the towel and the crunchy root on the plate",
            "move the polished one onto the towel and the skinny root onto the plate",
            "put the reflective one on the towel and the bright root on the round plate",
        ),
        "order": (
            "the spoon must sit on the towel and the carrot on the plate",
            "first the spoon onto the towel, then the carrot onto the plate",
            "have the spoon resting on the towel and the carrot resting on the plate",
            "the spoon is to be put on the towel and the carrot on the plate",
            "leave the spoon on the towel, then leave the carrot on the plate",
            "the spoon belongs on the towel and the carrot belongs on the plate",
            "I want the spoon lying on the towel and the carrot lying on the plate",
            "make sure the spoon rests on the towel and the carrot rests on the plate",
        ),
        "reasoning": (
            "put what you would stir coffee with on the towel and the root often roasted with honey on the plate",
            "put the one used to scoop sugar on the towel and the root a snowman wears as a nose on the plate",
            "put what you would measure salt with on the towel and the root shredded into coleslaw on the plate",
            "put what you stir cocoa with on the towel and the root rich in beta-carotene on the plate",
            "put the spoon on what you dry your hands with and the carrot on what you serve cake on",
            "put what you would dip into honey on the towel and the root you cut into sticks for hummus on the plate",
            "put the one used to stir a latte on the towel and the root pickled for banh mi on the plate",
            "put what you would scoop flour with on the towel and the root that grows underground on the plate",
        ),
    },
    "CarrotOnNearPlate-v0": {
        "synonyms": (
            "lay the carrot on the plate nearest the camera",
            "move the carrot onto the plate nearer to the camera",
            "rest the carrot on the platter closest to the camera",
            "transfer the carrot to the plate that sits nearer the camera",
            "put the carrot onto the saucer that is nearer to the camera",
            "lower the carrot onto the plate with less distance to the camera",
            "lay the carrot down on the nearer plate as seen from the camera",
            "shift the carrot onto the platter nearest to the camera",
        ),
        "attributes": (
            "put the tapered root on the round plate nearest the camera",
            "put the pointed root on the plate closer to the camera",
            "put the slender root on the plate in the foreground",
            "put the long thin root on the plate closest to the camera",
            "lay the cone-shaped root on the plate nearer the camera",
            "put the skinny root on the round plate in the foreground",
            "put the bright root onto the plate nearest to the camera",
            "put the crunchy root on the plate that is nearer to the camera",
        ),
        "order": (
            "on the plate closer to the camera, lay the carrot",
            "the carrot must sit on the plate nearest the camera",
            "the plate nearest the camera needs the carrot on it",
            "have the carrot resting on the plate closer to the camera",
            "the carrot is to be put on the plate nearer the camera",
            "grab the carrot and lay it on the plate closest to the camera",
            "it is the plate nearer to the camera that the carrot belongs on",
            "leave the carrot lying on the plate in the foreground",
        ),
        "reasoning": (
            "put the root a snowman wears as a nose on the plate closer to the camera",
            "put the root rich in beta-carotene on the plate nearest the camera",
            "put the root shredded into coleslaw on the plate nearer to the camera",
            "put the root you would juice for breakfast on the plate closest to the camera",
            "put the root that is said to help night vision on the plate in the foreground",
            "put the root often glazed with butter on the plate nearest to the camera",
            "put the root that grows underground with a leafy top on the plate closer to the camera",
            "put the root sliced into coins for a stir-fry on the plate nearer the camera",
        ),
    },
    "CarrotOnFarPlate-v0": {
        "synonyms": (
            "lay the carrot on the plate farthest from the camera",
            "move the carrot onto the plate further from the camera",
            "rest the carrot on the platter farthest from the camera",
            "transfer the carrot to the plate that sits farther from the camera",
            "put the carrot onto the saucer that is further from the camera",
            "lower the carrot onto the plate with more distance to the camera",
            "lay the carrot down on the more distant plate as seen from the camera",
            "shift the carrot onto the platter furthest from the camera",
        ),
        "attributes": (
            "put the tapered root on the round plate farthest from the camera",
            "put the pointed root on the plate farther from the camera",
            "put the slender root on the plate in the background",
            "put the long thin root on the plate furthest from the camera",
            "lay the cone-shaped root on the plate farther away from the camera",
            "put the skinny root on the round plate in the background",
            "put the bright root onto the plate further away from the camera",
            "put the crunchy root on the plate that is more distant from the camera",
        ),
        "order": (
            "on the plate farther from the camera, lay the carrot",
            "the carrot must sit on the plate farthest from the camera",
            "the plate farthest from the camera needs the carrot on it",
            "have the carrot resting on the plate farther from the camera",
            "the carrot is to be put on the plate further from the camera",
            "grab the carrot and lay it on the plate furthest from the camera",
            "it is the plate farther from the camera that the carrot belongs on",
            "leave the carrot lying on the plate in the background",
        ),
        "reasoning": (
            "put the root a snowman wears as a nose on the plate farther from the camera",
            "put the root rich in beta-carotene on the plate farthest from the camera",
            "put the root shredded into coleslaw on the plate further from the camera",
            "put the root you would juice for breakfast on the plate furthest from the camera",
            "put the root that is said to help night vision on the plate in the background",
            "put the root often glazed with butter on the plate farthest away from the camera",
            "put the root that grows underground with a leafy top on the plate farther from the camera",
            "put the root sliced into coins for a stir-fry on the plate further away from the camera",
        ),
    },
}


def rewording(env_name: str, family: str, episode_key: str) -> str:
    """The training rewording of `family` for an episode of the task `env_name`: one of the task's, always the same
    for the same episode (chosen by a checksum of the episode key)."""
    options = REWORDINGS[env_name][family]
    return options[zlib.crc32(f"{episode_key}|{family}".encode()) % len(options)]
