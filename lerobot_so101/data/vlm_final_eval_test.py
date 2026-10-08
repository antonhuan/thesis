"""
Final-evaluation tester for the orchestrator's high-level task judge.

Feeds real final-scene frames (the final_eval_*.png the orchestrator saved under
runs/) to the SAME evaluation path the robot uses -- i.e. what
_run_high_level_task_body does once the sub-task queue empties:

    planner.evaluate_final(prompt, final_frame)
        pass 1  survey_scene   image only (no instruction) -> object_locations rows
        pass 2  _judge_final   instruction + surveyed rows (no image) -> {success, reason}
        (fallback) single-call FINAL_EVALUATION_SYSTEM_PROMPT when the survey is empty

No evaluation logic is duplicated here: the planner is called as-is and
generate() is wrapped only to record every raw pass output. The survey is also
passed through orch.remaining_from_survey, which is what a retry round plans from.

Each frame is hand-labelled with where banana / toy / pouch rest
(vlm_final_eval_scenes.json). Every frame is crossed with every preference prompt
from vlm_decompose_test.PROMPTS, and the expected verdict is derived from the
labels, so one frame yields both positive and negative cases -- e.g. a frame with
the toy and pouch on the tray is a success for "everything except the banana" and
a failure for "put everything on the tray".

Expected verdict: the set of objects on the tray must equal the prompt's expected
set (quantity: exactly n objects from the pool). Order cannot be seen in a still,
so sequencing prompts are judged on the set alone -- the check is that the judge
does not fail a correct end state over an order it cannot observe. A case whose
verdict hinges on a "partial" object (straddling the rim) is not scored; a prompt
naming an object absent from the scene is skipped.

Per case:
    correct          final verdict == expected verdict
    survey_ok        every present, non-partial object surveyed at its true location
    judge_consistent verdict == what the survey's own rows imply (judge alone)
    retry_ok         remaining_from_survey names exactly the objects not on the tray
    cause            for a wrong verdict: single_call | perception | judge

Usage (inside the lerobot-so101 container, cwd /workspace):
    python vlm_final_eval_test.py                                  # all scenes x all prompts
    python vlm_final_eval_test.py --categories base,omission --trials 2
    python vlm_final_eval_test.py --scenes 2026-08-19_04-44-51_01 2026-09-16_08-31-16_02
    python vlm_final_eval_test.py --initial                        # BEFORE/AFTER survey prototype
    python vlm_final_eval_test.py --initial --sample 200           # fixed ~200-case subset
    python vlm_final_eval_test.py --initial --inventory --sample 200   # + monitor object grounder
    python vlm_final_eval_test.py --initial --sets-judge --looks --sample 200   # current best + looks
    python vlm_final_eval_test.py --rejudge vlm_final_eval_test/<stamp>/results.json   # judge only
    python vlm_final_eval_test.py --replot vlm_final_eval_test/<stamp>/results.json

Outputs (vlm_final_eval_test/<stamp>/):
    results.json     per-case log: raw pass outputs, survey rows, verdict, scores
    summary.md       per-prompt / per-category / per-scene tables + error attribution
    final_eval_results.png
"""

import argparse
import collections
import json
import logging
import random
import re
import time
from collections import OrderedDict
from datetime import datetime
from itertools import product
from pathlib import Path

from PIL import Image

from vlm_decompose_test import (CATEGORIES, DEFAULT_MODEL, FIXTURES, OBJECTS, PROMPTS, SYNONYMS,
                                canonical)

DEFAULT_SCENES = "vlm_final_eval_scenes.json"
CAUSES = OrderedDict([
    ("single_call", "survey came back empty; the single-call fallback judged the image"),
    ("perception", "the survey mislocated, missed or merged an object"),
    ("judge", "the survey was right; the judge misread the instruction against it"),
    ("no_verdict", "the evaluation output did not parse; the pipeline would end the task"),
])


# --------------------------------------------------------------------------- #
# Expected verdicts
# --------------------------------------------------------------------------- #
def _verdict(spec: dict, locs: dict[str, str]) -> bool | None:
    """Success of `spec` given unambiguous locations (tray / table / absent)."""
    present = {o for o, l in locs.items() if l != "absent"}
    on_tray = {o for o, l in locs.items() if l == "tray"}
    if spec.get("n"):
        pool = set(spec["pool"]) & present
        if len(pool) < spec["n"]:
            return None  # the instruction cannot be met in this scene
        return on_tray <= pool and len(on_tray) == spec["n"]
    return on_tray == set(spec["expected"]) & present


def expected_success(spec: dict, locs: dict[str, str]) -> bool | None:
    """Expected verdict, or None when the case cannot be scored.

    Objects labelled partial/unknown are tried both on and off the tray; if the
    verdict flips, it hinges on that object and the case is not scored.
    """
    if any(re.search(rf"\b{w}", spec["prompt"].lower())
           for o, l in locs.items() if l == "absent" for w in SYNONYMS[o]):
        return None  # e.g. "except the banana" in a scene with no banana
    unsure = [o for o, l in locs.items() if l in ("partial", "unknown")]
    outcomes = set()
    for combo in product(["tray", "table"], repeat=len(unsure)):
        outcomes.add(_verdict(spec, {**locs, **dict(zip(unsure, combo))}))
    return outcomes.pop() if len(outcomes) == 1 else None


# --------------------------------------------------------------------------- #
# Survey scoring
# --------------------------------------------------------------------------- #
def _is_fixture(name: str) -> bool:
    return any(re.search(rf"\b{f}", name.lower()) for f in FIXTURES + ["hand"])


def is_tray_edge(support: str) -> bool:
    """A survey support saying the tray holds the object up only partly (--tray-edge)."""
    sup = support.strip().lower()
    return "tray" in sup and bool(re.search(r"\b(edge|rim|partly|partially)\b", sup))


def edge_to(policy: str) -> str:
    """Where a tray-edge object counts: on the tray (loose) or not (strict)."""
    return "tray" if policy == "loose" else "table"


def survey_locations(rows: list[dict],
                     aliases: dict[str, str] | None = None,
                     policy: str = "loose") -> tuple[dict[str, str], list[str]]:
    """Survey rows -> ({object: tray/table/unknown}, fixture rows it emitted).

    Names map through the decompose test's synonyms; an object resting on another
    object takes that object's location (a toy on the pouch in the tray is on the
    tray). An object the survey never names is "absent" as far as the judge knows.
    `aliases` maps inventory names the synonyms miss ("sausage") to an object.
    """
    aliases = aliases or {}

    def to_obj(text: str) -> str | None:
        return canonical(text) or aliases.get(text.strip().lower())

    support, fixtures = {}, []
    for r in rows:
        name = str(r.get("object", ""))
        obj = to_obj(name)
        if obj is None:
            if _is_fixture(name):
                fixtures.append(name)
            continue
        # A repeated object (the change survey sometimes lists BEFORE rows, then AFTER
        # rows) takes its last row -- the AFTER one.
        support[obj] = str(r.get("support", "")).lower()

    def resolve(obj: str, seen: frozenset = frozenset()) -> str:
        sup = support[obj]
        if is_tray_edge(sup):
            return edge_to(policy)
        if "tray" in sup:
            return "tray"
        under = to_obj(sup)
        if under in support and under not in seen | {obj}:
            return resolve(under, seen | {obj})
        return "table" if "table" in sup else "unknown"

    return {o: resolve(o) if o in support else "absent" for o in OBJECTS}, fixtures


def score_case(spec: dict, scene: dict, surveyed: list[dict] | None, success: bool | None,
               remaining: list[str] | None, policy: str | None = None) -> dict:
    """`policy` (a --tray-edge run's tray policy) makes "partial" labels scorable: on the
    tray under loose, off it under strict. Without it they stay ambiguous, as before."""
    truth = scene["locations"]
    if policy:
        truth = {o: edge_to(policy) if l == "partial" else l for o, l in truth.items()}
    expected = expected_success(spec, truth)
    s = dict(expected=expected, correct=None if expected is None
             else success is not None and success == expected)
    if surveyed:
        seen, fixtures = survey_locations(surveyed, scene.get("aliases"), policy or "loose")
        checkable = [o for o, l in truth.items() if l not in ("partial", "absent")]
        s["survey_locations"] = seen
        s["survey_fixture_rows"] = fixtures
        s["survey_ok"] = all(seen[o] == truth[o] for o in checkable)
        s["survey_wrong"] = [o for o in checkable if seen[o] != truth[o]]
        implied = expected_success(spec, seen)
        s["judge_consistent"] = None if implied is None or success is None else success == implied
        if remaining is not None and "partial" not in truth.values():
            aliases = scene.get("aliases") or {}
            left = {canonical(n) or aliases.get(n.lower()) for n in remaining} - {None}
            s["retry_ok"] = left == {o for o, l in truth.items() if l == "table"}
    if s["correct"] is False:
        s["cause"] = ("no_verdict" if success is None
                      else "single_call" if not surveyed
                      else "perception" if not s["survey_ok"] else "judge")
    return s


# --------------------------------------------------------------------------- #
# --initial: BEFORE/AFTER variant of the survey -> judge split (test only)
# --------------------------------------------------------------------------- #
# Prototype, not yet in the orchestrator. The survey sees the scene before the robot
# started (the round-1 decompose frame) next to the final frame, still without the
# instruction, and reports where each object started and where it is now. The judge
# stays text-only and gets each object's "before" alongside its current "support".
CHANGE_SURVEY_SYSTEM_PROMPT = """You are a perception module for an orange tabletop robot arm (SO-101, 6-DOF). You receive two camera observations of the same workspace from the same camera: BEFORE, taken before the robot started, and AFTER, taken now. Your ONLY job is to list every object and report what it was RESTING ON in BEFORE and what it is RESTING ON in AFTER.

You MUST respond in the following JSON format exactly:

{
  "object_locations": [
    {"object": "<name>", "before": "<what it rested on in BEFORE>", "evidence": "<what you see in AFTER>", "support": "<what it rests on in AFTER>"}
  ]
}

What counts as an object:
- Find the objects in BEFORE first: they are usually spread out there and easiest to tell apart. Every object in BEFORE gets EXACTLY ONE row, under the name you gave it in BEFORE. Add a row for any object that appears only in AFTER.
- Surfaces and containers — the table, the tray, a bowl, a box — are SUPPORTS, not objects. Name them in "before" and "support", but never give them a row of their own.
- The orange robot arm is part of the robot, not an object. Do not give it a row. Ignore any text or label printed on it; that is not an object either.
- STACKED AND TOUCHING OBJECTS: an object resting ON TOP of another object is a separate object. Give it a row AND give the object beneath it a row. Never merge a stack into one entry.
- Use simple, concrete names for what you see (e.g. "cup", "banana", "pouch"). If you are unsure what something is, name it by its most obvious visual feature (colour or shape).

How to report where it is in AFTER:
- Fill in "evidence" BEFORE "support". Describe where the object actually sits in AFTER relative to the tray — inside its rim, touching its edge, or clear of it on the bare table, and roughly which side. Then, and only then, name the support.
- An object is on the tray ONLY if it lies inside the tray's rim. An object that is beside the tray, in front of it, behind it, or merely near it is on the table, not on the tray.
- If an object rests on top of another object rather than directly on a surface, name that object as its support.
- An object from BEFORE that you cannot see in AFTER has most likely ended up underneath another object. Name the object it is most likely hidden under as its support, and say in "evidence" that it is hidden. Only if it cannot be under anything, use "missing" as its support.
- Report what you SEE. Do not guess where an object ought to be, and do not assume anything has been moved, tidied, or put away.

Examples:
{"object_locations": [{"object": "apple", "before": "table", "evidence": "sitting inside the tray's rim, in the middle of the tray", "support": "tray"}, {"object": "banana", "before": "table", "evidence": "on the bare table to the left of the tray, not touching it", "support": "table"}]}
A pouch that was on the table and is now in the tray under a plush toy is two rows, and the toy rests on the pouch:
{"object_locations": [{"object": "pouch", "before": "table", "evidence": "inside the tray's rim, partly covered by the plush toy", "support": "tray"}, {"object": "plush toy", "before": "table", "evidence": "lying on top of the pouch, inside the tray", "support": "pouch"}]}
"""

def _with_tray_edge(prompt: str) -> str:
    """The change survey with a three-way tray answer (--tray-edge): whether the tray holds
    the object up fully, partly, or not at all. Code decides what "partly" counts as."""
    old = ("- An object is on the tray ONLY if it lies inside the tray's rim. An object that is "
           "beside the tray, in front of it, behind it, or merely near it is on the table, not on "
           "the tray.")
    new = ('- Decide what holds the object up. For an object resting on a surface, use exactly '
           'one of these supports:\n'
           '  - "tray": the object lies inside the tray\'s rim.\n'
           '  - "tray edge": the tray holds part of the object up — it rests partly on the tray or '
           'on its rim, with part of it hanging over the edge.\n'
           '  - "table": the tray holds none of it up — it lies on the table beside, in front of, '
           'behind or near the tray, even if it touches the tray\'s side.')
    assert prompt.count(old) == 1, "change survey prompt layout changed"
    out = prompt.replace(old, new)
    ex = ('A banana lying across the tray\'s rim, half inside and half hanging over the table, '
          'rests on the tray edge:\n'
          '{"object_locations": [{"object": "banana", "before": "table", "evidence": "across the '
          'tray\'s front rim, one end inside the tray and the other over the table", "support": '
          '"tray edge"}]}\n')
    return out.rstrip("\n") + "\n" + ex


def _with_looks(prompt: str) -> str:
    """The change survey plus a location-free "looks" field per object (--looks)."""
    out = prompt.replace('{"object": "<name>", "before":',
                         '{"object": "<name>", "looks": "<colour, shape, material>", "before":')
    out = out.replace(
        "- Use simple, concrete names for what you see",
        '- "looks" describes the object ITSELF in a few words — colour, shape, material (e.g. '
        '"brown, furry plush", "yellow curved fruit"). Never put where it is, or what it rests on, in "looks".\n'
        "- Use simple, concrete names for what you see")
    out = out.replace('{"object": "apple", "before"', '{"object": "apple", "looks": "red, round fruit", "before"')
    out = out.replace('{"object": "banana", "before"', '{"object": "banana", "looks": "yellow, curved fruit", "before"')
    out = out.replace('{"object": "pouch", "before"', '{"object": "pouch", "looks": "brown, zipped fabric case", "before"')
    out = out.replace('{"object": "plush toy", "before"', '{"object": "plush toy", "looks": "brown, furry plush", "before"')
    assert out.count('"looks"') >= 6, "change survey prompt layout changed"
    return out


_JUDGE_BEFORE_NOTE = """Each finding has "before": what the object rested on BEFORE the robot started, and "support": where the object is NOW. "support" is already resolved through stacks: "tray (on top of the banana)" means the object IS on the tray — it rests on the banana, which is in the tray. Judge every object by the location before the brackets; the brackets only describe how it sits.
- An object the instruction asked to leave, skip, keep, or not touch must still NOT be on the tray.
- A "support" of "missing" means the object was in the scene before but cannot be found now: it is NOT at its destination.

"""


_JUDGE_LOOKS_NOTE = """Each finding also has "looks": what the object looks like (colour, shape, material). Use it when the instruction picks objects by colour, shape, material or kind.

"""

# --inventory: the monitor's object grounder (vlm_monitor_tune.scene_inventory) builds the
# object list once from the BEFORE frame; the survey reports a row for exactly those
# names, and the judge gets each object's appearance with its name.
INVENTORY_SURVEY_SYSTEM_PROMPT = """You are a perception module for an orange tabletop robot arm (SO-101, 6-DOF). You receive two camera observations of the same workspace from the same camera — BEFORE, taken before the robot started, and AFTER, taken now — plus the list of objects that were on the table in BEFORE, each with a short description of what it looks like. Your ONLY job is to report, for every listed object, what it rested on in BEFORE and what it is RESTING ON in AFTER.

You MUST respond in the following JSON format exactly:

{
  "object_locations": [
    {"object": "<name exactly as listed>", "before": "<what it rested on in BEFORE>", "evidence": "<what you see in AFTER>", "support": "<what it rests on in AFTER>"}
  ]
}

Objects:
- Give EXACTLY ONE row to every listed object, using its name EXACTLY as listed. Use the descriptions to find each object in AFTER — once moved it may be turned, partly covered, or lying on something else.
- If you see an object in AFTER that is not on the list, add a row for it with a short name of your own.
- Surfaces and containers — the table, the tray — are SUPPORTS, not objects. Never give them a row. The orange robot arm is not an object either.
- STACKED AND TOUCHING OBJECTS: an object resting ON TOP of another listed object names that object as its support. Both objects keep their own rows.

How to report where it is in AFTER:
- Fill in "evidence" BEFORE "support". Describe where the object actually sits in AFTER relative to the tray — inside its rim, touching its edge, or clear of it on the bare table, and roughly which side. Then, and only then, name the support.
- An object is on the tray ONLY if it lies inside the tray's rim. An object that is beside the tray, in front of it, behind it, or merely near it is on the table, not on the tray.
- A listed object you cannot see in AFTER has most likely ended up underneath another object. Name the object it is most likely hidden under as its support, and say in "evidence" that it is hidden. Only if it cannot be under anything, use "missing" as its support.
- Report what you SEE. Do not guess where an object ought to be, and do not assume anything has been moved, tidied, or put away.

Example, for the list "apple: red, round fruit" and "cloth: blue, folded fabric":
{"object_locations": [{"object": "apple", "before": "table", "evidence": "inside the tray's rim, lying on the blue cloth", "support": "cloth"}, {"object": "cloth", "before": "table", "evidence": "spread inside the tray's rim, under the apple", "support": "tray"}]}
"""


def resolve_findings(rows: list[dict], looks: dict[str, str] | None = None,
                     policy: str = "loose") -> list[dict]:
    """Survey rows -> judge findings with each object's support resolved to where it ends up.

    A support naming another surveyed object (by the same, a shorter or a longer
    name, as in orch.remaining_from_survey) is followed down the stack to a surface,
    so the toy on the banana in the tray becomes "tray (on top of the banana)". The
    judge no longer has to follow the chain itself.
    """
    names = [str(r.get("object", "")).strip() for r in rows]
    support = {n.lower(): str(r.get("support", "")).strip().lower()
               for n, r in zip(names, rows) if n}  # duplicate rows: the later (AFTER) one wins

    def as_object(sup: str, me: str) -> str | None:
        if not sup or "tray" in sup or "table" in sup:
            return None  # a surface, never another object
        if sup in support and sup != me:
            return sup
        hits = [k for k in support if k != me and (sup in k or k in sup)]
        return max(hits, key=len) if hits else None

    def settle(key: str) -> str:
        seen = {key}
        while (under := as_object(support[key], key)) and under not in seen:
            seen.add(under)
            key = under
        sup = support[key]
        if is_tray_edge(sup):
            return edge_to(policy)
        return "tray" if "tray" in sup else "table" if "table" in sup else sup or "unknown"

    findings, last = [], {n.lower(): i for i, n in enumerate(names) if n}
    for i, (n, r) in enumerate(zip(names, rows)):
        if not n or last[n.lower()] != i:
            continue  # one finding per object: its last (AFTER) row
        under = as_object(support[n.lower()], n.lower())
        how = "hidden under" if "hidden" in str(r.get("evidence", "")).lower() else "on top of"
        notes = [f"{how} the {under}"] if under else []
        if is_tray_edge(support[n.lower()]):
            notes.append("partly on the tray edge")
        finding = {"object": n, "before": str(r.get("before")),
                   "support": settle(n.lower()) + (f" ({'; '.join(notes)})" if notes else "")}
        if looks is not None:
            finding["looks"] = looks.get(n.lower(), "")
        findings.append(finding)
    return findings


# --strict-judge: three additions from the 0.79 run's judge errors -- order is
# unobservable, the tray set must match exactly (written out before the verdict), and
# findings may name an object differently from the instruction. Replaces the base
# prompt's output format, since the judge now returns its required sets first.
_STRICT_READING = """Reading the instruction:
- ORDER CANNOT BE SEEN. Words about order — first, then, next, last, after, before, starting with — only describe the order the robot placed things in. One final scene cannot show order, so ignore order completely and judge only where each object ends up. "Put everything on the tray, but the pouch goes last" means every object, the pouch included, must end on the tray.
- EXACT SET. Every object the instruction asks to move must be on the tray, and every other object must NOT be on the tray — an object the instruction does not ask to move should have been left alone. "Only", "except", "but not", "leave", "don't touch" and conditions ("if there is ...") decide which objects belong in the set. A number means exactly that many: "put one object on the tray" is failed when two objects are on the tray.
- NAMES DIFFER. The findings come from a separate perception module and may name an object differently from the instruction — a "brown cylinder" or "brown roll" may be the pouch, a "stuffed animal" may be the toy. Match each object the instruction mentions to the finding that is plausibly the same thing. Never conclude that an object is missing when one of the findings could be it.

"""

_STRICT_OUTPUT = """Output format:
Return ONLY a JSON object, with the fields in this order:

{"must_be_on_tray": ["<finding name>", ...], "must_not_be_on_tray": ["<finding name>", ...], "choose_from": ["<finding name>", ...], "choose_count": <number>, "reason": "<one or two sentences>", "success": true or false}

Rules:
- Decide the lists from the instruction alone, before looking at where the objects are. Use finding names exactly as written, and put EVERY finding in exactly one of must_be_on_tray, must_not_be_on_tray or choose_from.
- choose_from / choose_count are only for instructions that let the robot choose ("put one object on the tray", "put two of the brown objects on the tray"): list the objects it may choose from and how many. Otherwise leave choose_from empty and choose_count 0.
- "success": true only if every object in must_be_on_tray is on the tray, no object in must_not_be_on_tray is on the tray, and exactly choose_count objects of choose_from are on the tray.
- "reason": if success is false, name the specific object(s) that are not where the instruction requires, and where the findings say they actually are. State a conclusion only — do NOT deliberate, reconsider, re-read the instruction, or revise yourself inside this field. Never write "wait", "actually", or "I made a mistake".

Examples:

Instruction: "put everything on the tray except for the banana"
Findings: [{"object": "apple", "before": "table", "support": "tray"}, {"object": "banana", "before": "table", "support": "table"}]
{"must_be_on_tray": ["apple"], "must_not_be_on_tray": ["banana"], "choose_from": [], "choose_count": 0, "reason": "The apple is on the tray and the banana was correctly left on the table.", "success": true}

Instruction: "put one of the red objects on the tray"
Findings: [{"object": "apple", "before": "table", "support": "tray"}, {"object": "red cup", "before": "table", "support": "tray (on top of the apple)"}, {"object": "banana", "before": "table", "support": "table"}]
{"must_be_on_tray": [], "must_not_be_on_tray": ["banana"], "choose_from": ["apple", "red cup"], "choose_count": 1, "reason": "Both red objects, the apple and the red cup, are on the tray, but only one was asked for.", "success": false}

Instruction: "put the cup on the tray first, then everything else"
Findings: [{"object": "blue mug", "before": "table", "support": "tray"}, {"object": "spoon", "before": "table", "support": "tray (on top of the blue mug)"}]
{"must_be_on_tray": ["blue mug", "spoon"], "must_not_be_on_tray": [], "choose_from": [], "choose_count": 0, "reason": "The cup (the blue mug) and the spoon are both on the tray; the order cannot be judged from the final scene.", "success": true}
"""


def change_judge_prompt(base: str, looks: bool = False, strict: bool = False) -> str:
    assert base.count("Output format:") == 1, "FINAL_JUDGE_SYSTEM_PROMPT layout changed"
    note = _JUDGE_BEFORE_NOTE + (_JUDGE_LOOKS_NOTE if looks else "")
    if strict:
        return base.split("Output format:")[0] + note + _STRICT_READING + _STRICT_OUTPUT
    return base.replace("Output format:", note + "Output format:")


# --sets-judge: the requirement is decided by a call that never sees where anything is
# (only the instruction + the object names), and code compares it with the resolved
# locations. The strict judge showed why: given locations, it fits its "required" lists
# to the scene.
REQUIREMENTS_SYSTEM_PROMPT = """You decide what a tabletop robot instruction requires. You receive the instruction and the names of the objects on the table. You do NOT see the scene and are not told where anything is — decide only what the instruction asks for.

The tray is the destination: "the tray" and "away" always mean the tray. The robot arm, the table and the tray are not objects.

Sort every listed object into exactly one group:
- must_be_on_tray: the instruction asks for it to end up on the tray.
- must_not_be_on_tray: the instruction asks to leave it, skip it, keep it or not touch it — or simply does not ask for it to be moved. An object the instruction does not ask to move must stay off the tray.
- choose_from, with choose_count: only when the instruction lets the robot choose ("put one object on the tray", "put two of the brown objects on the tray") — the objects it may choose from, and exactly how many it must place.

Rules:
- Names may differ from the instruction's words: the instruction's "pouch" may be listed as a "brown cylinder" or "brown roll", its "toy" as a "stuffed animal". Match by what each object plausibly is. For words about colour, material or kind ("brown", "soft", "edible", "fruit"), decide from what each named object is.
- Words about order — first, then, next, last, after, starting with — only describe the order of placement. Ignore order: an object that "goes last" is placed after the others, so it must still end up on the tray.
- Conditions ("if there is an apple on the table ...") are decided from the list: an object that is not listed is not on the table.
- Use the names exactly as listed, and put every listed object in exactly one group.

Return ONLY JSON:
{"must_be_on_tray": ["<name>", ...], "must_not_be_on_tray": ["<name>", ...], "choose_from": ["<name>", ...], "choose_count": <number>}

Examples, for the objects ["apple", "red cup", "banana"]:
Instruction: "put everything on the tray except the banana"
{"must_be_on_tray": ["apple", "red cup"], "must_not_be_on_tray": ["banana"], "choose_from": [], "choose_count": 0}
Instruction: "put one of the red objects on the tray"
{"must_be_on_tray": [], "must_not_be_on_tray": ["banana"], "choose_from": ["apple", "red cup"], "choose_count": 1}
Instruction: "put the fruit on the tray, the banana first"
{"must_be_on_tray": ["apple", "banana"], "must_not_be_on_tray": ["red cup"], "choose_from": [], "choose_count": 0}
Instruction: "put everything on the tray, but the red cup goes last"
{"must_be_on_tray": ["apple", "red cup", "banana"], "must_not_be_on_tray": [], "choose_from": [], "choose_count": 0}
"""


def _json_object(text: str) -> dict:
    text = re.sub(r"^```\w*\s*|\s*```$", "", text.strip())
    for cand in (text, text[text.find("{"): text.rfind("}") + 1]):
        try:
            out = json.loads(cand)
            if isinstance(out, dict):
                return out
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse requirements JSON:\n{text}")


def compare_sets(req: dict, findings: list[dict]) -> dict:
    """Code verdict: the requirement's sets against the resolved locations."""
    names = [f["object"] for f in findings]
    where = {f["object"].lower(): f["support"].split(" (")[0] for f in findings}

    def snap(n: str) -> str | None:
        n = str(n).strip().lower()
        if n in where:
            return n
        hits = [k for k in where if n and (n in k or k in n)]
        return max(hits, key=len) if hits else None

    groups, unmatched = {}, []
    for key in ("must_be_on_tray", "must_not_be_on_tray", "choose_from"):
        vals = req.get(key) or []
        if not isinstance(vals, list):
            raise ValueError(f"requirements field {key} is not a list: {vals!r}")
        groups[key] = []
        for v in vals:
            k = snap(v)
            (groups[key].append(k) if k else unmatched.append(str(v)))
    # For "put one/two objects" the call tends to list every object under both
    # choose_from and must_not_be_on_tray; the positive group wins.
    allowed = set(groups["must_be_on_tray"]) | set(groups["choose_from"])
    groups["must_not_be_on_tray"] = [k for k in groups["must_not_be_on_tray"] if k not in allowed]
    listed = {k for g in groups.values() for k in g}
    unlisted = [n for n in names if n.lower() not in listed]
    groups["must_not_be_on_tray"] += [n.lower() for n in unlisted]  # not asked -> leave it
    count = int(req.get("choose_count") or 0)

    problems = [f"{k} should be on the tray (now: {where[k]})"
                for k in groups["must_be_on_tray"] if where[k] != "tray"]
    problems += [f"{k} should not be on the tray" for k in groups["must_not_be_on_tray"]
                 if where[k] == "tray"]
    if groups["choose_from"]:
        placed = [k for k in groups["choose_from"] if where[k] == "tray"]
        if len(placed) != count:
            problems.append(f"{len(placed)} of {groups['choose_from']} on the tray, expected exactly {count}")
    return {"success": not problems,
            "reason": "; ".join(problems) or "every object is where the instruction requires",
            "judge_sets": {**groups, "choose_count": count, "unlisted": unlisted, "unmatched": unmatched}}


def sets_judge(planner, orch, prompt: str, surveyed: list[dict], before=None,
               policy: str = "loose") -> dict:
    """Requirements from the instruction + object names only, then a code comparison.

    With `before` (--sets-image) the requirements call also sees the BEFORE frame, to
    tell what each object is (colour, material, kind). It shows the start of the task,
    so it says nothing about where objects ended up.
    """
    findings = resolve_findings(surveyed, policy=policy)
    looks = {str(r.get("object", "")).strip().lower(): str(r.get("looks", "")).strip()
             for r in surveyed if r.get("looks")}  # last row per object, as in resolve_findings
    if looks:
        listed = "\n".join(f'- "{f["object"]}": {looks.get(f["object"].lower(), "")}' for f in findings)
        text = (f"Objects on the table (name: what it looks like):\n{listed}"
                f'\nInstruction: "{prompt}"')
    else:
        text = (f"Objects on the table: {json.dumps([f['object'] for f in findings])}"
                f'\nInstruction: "{prompt}"')
    content = [{"type": "text", "text": text}]
    if before is not None:
        content = [{"type": "text", "text": "[BEFORE — the table before the robot started. Use it only "
                                            "to see what each listed object is (colour, material, kind); "
                                            "it does not show where anything is now.]"},
                   {"type": "image", "image": before}] + content
    out = orch.generate(planner.model, planner.processor, [
        {"role": "system", "content": [{"type": "text", "text": REQUIREMENTS_SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ], temperature=planner.temperature)
    req = _json_object(out)
    verdict = compare_sets(req, findings)
    verdict.update(surveyed=surveyed, findings=findings, requirements=req)
    if any(is_tray_edge(str(r.get("support", ""))) for r in surveyed):
        # Same requirements under the other policy, so one run can be scored both ways.
        other = "strict" if policy == "loose" else "loose"
        verdict[f"success_{other}"] = compare_sets(
            req, resolve_findings(surveyed, policy=other))["success"]
    return verdict


def sets_imply(verdict: dict, findings: list[dict]) -> bool | None:
    """Diagnostic only: the success the strict judge's own lists imply for these findings."""
    if not isinstance(verdict.get("must_be_on_tray"), list):
        return None
    on = {f["object"].lower() for f in findings if f["support"].startswith("tray")}
    names = lambda k: {str(n).lower() for n in (verdict.get(k) or [])}
    try:
        count = int(verdict.get("choose_count") or 0)
    except (TypeError, ValueError):
        return None
    return (names("must_be_on_tray") <= on and not (names("must_not_be_on_tray") & on)
            and (not names("choose_from") or len(names("choose_from") & on) == count))


def build_inventory(planner, initial) -> dict:
    """The monitor's grounder on the BEFORE frame: pickable objects with appearance,
    robot/fixture entries split off exactly as the live monitor does."""
    import vlm_monitor_tune as T
    from vlm_monitor_live import split_fixtures
    raw = T.scene_inventory(planner.model, planner.processor, initial, planner.temperature)
    objects, dropped = split_fixtures(raw)
    return dict(objects=objects, dropped=dropped)


def inventory_aliases(objects: list[dict]) -> dict[str, str]:
    """Scoring only: inventory name -> banana/toy/pouch, by its name, else its appearance."""
    out = {}
    for o in objects:
        obj = canonical(o["name"]) or canonical(o.get("appearance", ""))
        if obj:
            out[o["name"].lower()] = obj
    return out


def evaluate_final_with_initial(planner, vlm, orch, prompt: str, frame, initial,
                                inventory: list[dict] | None = None, strict: bool = False,
                                sets: bool = False, looks: bool = False,
                                sets_image: bool = False, tray_edge: bool = False,
                                policy: str = "loose") -> dict:
    """evaluate_final with a BEFORE frame: change survey -> text-only judge.

    Mirrors VLMPlanner.evaluate_final / _judge_final, including the single-call
    fallback when the survey comes back empty.
    """
    content = [
        {"type": "text", "text": "[BEFORE — top-down camera, before the robot started]"},
        {"type": "image", "image": initial},
        {"type": "text", "text": "[AFTER — top-down camera, now]"},
        {"type": "image", "image": frame},
    ]
    if inventory:
        listed = "\n".join(f"- {o['name']}: {o['appearance']}" for o in inventory)
        content.append({"type": "text", "text": f"\nObjects on the table in BEFORE:\n{listed}\n\n"
                        "For every listed object, report what it rested on in BEFORE and what it "
                        "rests on in AFTER."})
        system = INVENTORY_SURVEY_SYSTEM_PROMPT
    else:
        content.append({"type": "text", "text": "\nList every object, what it rested on in BEFORE, "
                                                "and what it rests on in AFTER."})
        system = CHANGE_SURVEY_SYSTEM_PROMPT
        if tray_edge:
            system = _with_tray_edge(system)
        if looks:
            system = _with_looks(system)
    out = vlm.generate(planner.model, planner.processor, [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": content},
    ], temperature=planner.temperature)
    surveyed = vlm.parse_object_locations(out)
    if inventory:
        # Snap names back onto the inventory in code, as the monitor does (match_name).
        from vlm_monitor_tune import match_name
        names = [o["name"] for o in inventory]
        for r in surveyed:
            snapped = match_name(str(r.get("object", "")), names)
            if snapped and snapped != str(r.get("object", "")).strip().lower():
                r["raw_object"], r["object"] = r["object"], snapped
    if not surveyed:
        # Same single-call fallback evaluate_final uses when its survey is empty.
        logging.warning("Change survey returned no rows — single-call fallback.")
        text = (f'\nThe robot was given this high-level instruction: "{prompt}"'
                f'\nLooking at the final scene, was the whole instruction accomplished?')
        return orch.parse_evaluation(orch.generate(planner.model, planner.processor, [
            {"role": "system", "content": [{"type": "text",
                                            "text": orch.FINAL_EVALUATION_SYSTEM_PROMPT}]},
            {"role": "user", "content": planner._user_content(frame, text)},
        ], temperature=planner.temperature))
    if sets:
        return sets_judge(planner, orch, prompt, surveyed, initial if sets_image else None, policy)
    return judge_change_findings(planner, orch, prompt, surveyed, inventory, strict)


def judge_change_findings(planner, orch, prompt: str, surveyed: list[dict],
                          inventory: list[dict] | None = None, strict: bool = False) -> dict:
    """Text-only judge over the change survey's rows, stacks resolved in code first.
    With an inventory, each finding also carries the object's appearance."""
    looks = {o["name"].lower(): o["appearance"] for o in inventory} if inventory else None
    findings = resolve_findings(surveyed, looks)
    text = (f'The robot was given this high-level instruction: "{prompt}"'
            f"\n\nPerception findings for the final scene:"
            f"\n{json.dumps(findings)}"
            f"\n\nWas the whole instruction accomplished?")
    verdict = orch.parse_evaluation(orch.generate(planner.model, planner.processor, [
        {"role": "system", "content": [{"type": "text",
                                        "text": change_judge_prompt(orch.FINAL_JUDGE_SYSTEM_PROMPT,
                                                                    looks=bool(inventory),
                                                                    strict=strict)}]},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ], temperature=planner.temperature))
    verdict["surveyed"] = surveyed
    verdict["findings"] = findings
    if strict:
        verdict["judge_sets"] = {k: verdict.get(k) for k in
                                 ("must_be_on_tray", "must_not_be_on_tray", "choose_from", "choose_count")}
        verdict["sets_imply"] = sets_imply(verdict, findings)
    return verdict


def initial_frame_path(final_path: str) -> str:
    """The round-1 decompose frame in the same task folder (not decompose_roundN)."""
    found = sorted(Path(final_path).parent.glob("decompose_2*.png"))
    if not found:
        raise FileNotFoundError(f"No round-1 decompose frame next to {final_path}")
    return str(found[0])


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def sample_cases(scenes: list[dict], prompts: dict, n: int, seed: int = 0) -> list[tuple[dict, str]]:
    """A fixed, reviewable subset of scene x prompt cases (~n in total).

    Every prompt gets an equal share. Within a prompt, up to 45% of its share are
    cases whose task WAS done (expected success) -- the full cross product is ~18% --
    and the rest are failures. Each pick goes to the least-used scene so far, so the
    subset spreads over all frames. Same labels + prompts + seed -> same subset.
    """
    rng = random.Random(seed)
    use = collections.Counter()
    keys = list(prompts)
    quota = {k: n // len(keys) + (i < n % len(keys)) for i, k in enumerate(keys)}
    cases = []
    for key in keys:
        spec = prompts[key]
        pos = [sc for sc in scenes if expected_success(spec, sc["locations"]) is True]
        neg = [sc for sc in scenes if expected_success(spec, sc["locations"]) is False]
        n_pos = min(len(pos), round(quota[key] * 0.45))
        for pool, want in ((pos, n_pos), (neg, quota[key] - n_pos)):
            cand = pool[:]
            rng.shuffle(cand)
            for _ in range(min(want, len(cand))):
                pick = min(cand, key=lambda sc: use[sc["name"]])  # ties: shuffled order
                cand.remove(pick)
                use[pick["name"]] += 1
                cases.append((pick, key))
    return cases


def load_scenes(path: Path, names: list[str] | None) -> list[dict]:
    scenes = json.loads(path.read_text())["scenes"]
    if names:
        scenes = [s for s in scenes if s["name"] in names]
    return scenes


def install_recorder(vlm, orch, passes: list[dict]):
    """Wrap generate() in both modules to log every raw pass; returns an undo callable."""
    def recording(real, module_name):
        def wrapped(model, processor, messages, *a, **kw):
            system = messages[0]["content"][0]["text"]
            name = ("survey" if system == vlm.SCENE_SURVEY_SYSTEM_PROMPT
                    else "change_survey" if system in (CHANGE_SURVEY_SYSTEM_PROMPT,
                                                       _with_looks(CHANGE_SURVEY_SYSTEM_PROMPT),
                                                       _with_tray_edge(CHANGE_SURVEY_SYSTEM_PROMPT),
                                                       _with_looks(_with_tray_edge(CHANGE_SURVEY_SYSTEM_PROMPT)),
                                                       INVENTORY_SURVEY_SYSTEM_PROMPT)
                    else "judge" if system in [orch.FINAL_JUDGE_SYSTEM_PROMPT] + [
                        change_judge_prompt(orch.FINAL_JUDGE_SYSTEM_PROMPT, lk, st)
                        for lk in (False, True) for st in (False, True)]
                    else "requirements" if system == REQUIREMENTS_SYSTEM_PROMPT
                    else "single_call" if system == orch.FINAL_EVALUATION_SYSTEM_PROMPT
                    else f"other ({module_name})")
            t0 = time.time()
            out = real(model, processor, messages, *a, **kw)
            passes.append(dict(pass_name=name, output=out, latency_s=round(time.time() - t0, 2)))
            return out
        return wrapped

    # survey_scene calls vlm.generate; the judge / single-call call orch.generate.
    real_vlm, real_orch = vlm.generate, orch.generate
    vlm.generate, orch.generate = recording(real_vlm, "vlm"), recording(real_orch, "orch")

    def restore():
        vlm.generate, orch.generate = real_vlm, real_orch
    return restore


def rejudge(args) -> Path:
    """Re-run only the judge on an --initial run's saved change surveys (no images needed)."""
    import vlm
    import vlm_robot_orchestrator as orch

    src = Path(args.rejudge)
    results = json.loads(src.read_text())
    if not results["meta"].get("initial"):
        raise SystemExit("--rejudge needs a results.json from an --initial run")
    passes: list[dict] = []
    restore = install_recorder(vlm, orch, passes)
    m = results["meta"]
    planner = orch.VLMPlanner(m["model"], temperature=m["temperature"], two_pass=m["two_pass"])
    out_dir = Path(args.out_root) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    m.update(rejudge_of=str(src), strict_judge=args.strict_judge, sets_judge=args.sets_judge,
             sets_image=args.sets_image,
             created=datetime.now().isoformat(timespec="seconds"))
    ts = results["trials"]
    by_name = {sc["name"]: sc for sc in results["scenes"]}
    befores: dict = {}
    for i, t in enumerate(ts, 1):
        if not t["surveyed"]:
            continue  # single-call fallback: nothing to re-judge, keep its verdict
        passes.clear()
        survey_passes = [p for p in t["passes"] if p["pass_name"] == "change_survey"]
        error, verdict = None, {}
        try:
            inv = by_name[t["scene"]].get("inventory") if m.get("inventory") else None
            before = None
            if args.sets_image:
                path = initial_frame_path(by_name[t["scene"]]["path"])
                if path not in befores:
                    befores[path] = Image.open(path).convert("RGB")
                before = befores[path]
            verdict = (sets_judge(planner, orch, t["prompt"], t["surveyed"], before) if args.sets_judge
                       else judge_change_findings(planner, orch, t["prompt"], t["surveyed"],
                                                  inv["objects"] if inv else None, args.strict_judge))
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        t.update(success=bool(verdict["success"]) if "success" in verdict else None,
                 reason=verdict.get("reason", ""), error=error,
                 findings=verdict.get("findings"), judge_sets=verdict.get("judge_sets"),
                 sets_imply=verdict.get("sets_imply"), passes=survey_passes + list(passes))
        logging.info(f"[{i}/{len(ts)}] {t['scene']} {t['key']}: got={t['success']}")
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    restore()
    return out_dir / "results.json"


def run(args) -> Path:
    import vlm
    import vlm_robot_orchestrator as orch

    passes: list[dict] = []
    restore = install_recorder(vlm, orch, passes)

    out_dir = Path(args.out_root) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = load_scenes(Path(args.scenes_file), args.scenes)
    cats = args.categories.split(",") if args.categories else CATEGORIES
    prompts = OrderedDict((k, v) for k, v in PROMPTS.items() if v["category"] in cats)

    planner = orch.VLMPlanner(args.model, temperature=args.temp, two_pass=not args.single_call)
    results = dict(
        meta=dict(scenes_file=args.scenes_file, model=args.model, temperature=args.temp,
                  initial=args.initial, inventory=args.inventory, strict_judge=args.strict_judge,
                  sets_judge=args.sets_judge, looks=args.looks, sets_image=args.sets_image,
                  tray_edge=args.tray_edge, tray_policy=args.tray_policy if args.tray_edge else None,
                  orchestrator=args.orchestrator,
                  sample=args.sample, seed=args.seed,
                  two_pass=not args.single_call, trials=args.trials,
                  created=datetime.now().isoformat(timespec="seconds")),
        scenes=scenes, prompts=prompts, trials=[],
    )
    if args.sample:
        pairs = sample_cases(scenes, prompts, args.sample, args.seed)
    else:
        pairs = [(sc, k) for sc, k in product(scenes, prompts)
                 if expected_success(prompts[k], sc["locations"]) is not None or args.keep_unscored]
    cases = [(sc, k, t) for sc, k in pairs for t in range(args.trials)]
    frames = {sc["name"]: Image.open(sc["path"]).convert("RGB") for sc in scenes}
    initials = {}
    if args.initial:
        for sc in scenes:
            sc["initial_path"] = initial_frame_path(sc["path"])
            initials[sc["name"]] = Image.open(sc["initial_path"]).convert("RGB")
    if args.inventory:
        used = {sc["name"] for sc, _ in pairs}
        by_initial: dict[str, dict] = {}
        for sc in scenes:
            if sc["name"] not in used:
                continue
            if sc["initial_path"] not in by_initial:
                by_initial[sc["initial_path"]] = build_inventory(planner, initials[sc["name"]])
                inv = by_initial[sc["initial_path"]]
                logging.info(f"Inventory {sc['initial_path']}: "
                             + "; ".join(f"{o['name']} ({o['appearance']})" for o in inv["objects"])
                             + (f"  | dropped: {[o['name'] for o in inv['dropped']]}" if inv["dropped"] else ""))
            sc["inventory"] = by_initial[sc["initial_path"]]
            sc["aliases"] = inventory_aliases(sc["inventory"]["objects"])
    for i, (scene, key, t) in enumerate(cases, 1):
        spec = prompts[key]
        passes.clear()
        error, verdict = None, {}
        t0 = time.time()
        try:
            if args.orchestrator:
                # The ported path, exactly as the orchestrator runs it.
                verdict = planner.evaluate_final(spec["prompt"], frames[scene["name"]],
                                                 initials[scene["name"]])
            elif args.initial:
                verdict = evaluate_final_with_initial(
                    planner, vlm, orch, spec["prompt"], frames[scene["name"]], initials[scene["name"]],
                    scene["inventory"]["objects"] if args.inventory else None, args.strict_judge,
                    args.sets_judge, args.looks, args.sets_image, args.tray_edge, args.tray_policy)
            else:
                verdict = planner.evaluate_final(spec["prompt"], frames[scene["name"]])
        except Exception as e:  # unparseable output is a result, not a crash
            error = f"{type(e).__name__}: {e}"
        surveyed = verdict.get("surveyed") or []
        remaining = orch.remaining_from_survey(surveyed)[0] if surveyed else None
        success = bool(verdict["success"]) if "success" in verdict else None
        trial = dict(scene=scene["name"], key=key, category=spec["category"], prompt=spec["prompt"],
                     trial=t, success=success, reason=verdict.get("reason", ""), error=error,
                     surveyed=surveyed, findings=verdict.get("findings"),
                     judge_sets=verdict.get("judge_sets"), sets_imply=verdict.get("sets_imply"),
                     requirements=verdict.get("requirements"),
                     success_strict=verdict.get("success_strict"),
                     success_loose=verdict.get("success_loose"),
                     remaining=remaining,
                     latency_s=round(time.time() - t0, 2), passes=list(passes))
        trial["score"] = score_case(spec, scene, surveyed, success, remaining,
                                    args.tray_policy if args.tray_edge else None)
        results["trials"].append(trial)
        sc = trial["score"]
        logging.info(f"[{i}/{len(cases)}] {scene['name']} {key} #{t}: expected={sc['expected']} "
                     f"got={success} {'OK' if sc['correct'] else sc.get('cause', 'unscored')}")
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    restore()
    return out_dir / "results.json"


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _rate(ts: list[dict], field: str) -> tuple[float | None, int]:
    vals = [t["score"].get(field) for t in ts if t["score"].get(field) is not None]
    return (sum(vals) / len(vals) if vals else None), len(vals)


def _fmt(v):
    return "-" if v is None else f"{v:.2f}"


def group_stats(ts: list[dict]) -> dict:
    scored = [t for t in ts if t["score"]["correct"] is not None]
    pos = [t for t in scored if t["score"]["expected"]]
    neg = [t for t in scored if not t["score"]["expected"]]
    return dict(
        n=len(scored),
        acc=_rate(scored, "correct")[0],
        # false negative = task done but judged failed (triggers a needless retry round);
        # false positive = task not done but judged complete (the pipeline stops early).
        fn=sum(not t["success"] for t in pos), n_pos=len(pos),
        fp=sum(bool(t["success"]) for t in neg), n_neg=len(neg),
        survey=_rate(scored, "survey_ok")[0], judge=_rate(scored, "judge_consistent")[0],
        causes=collections.Counter(t["score"].get("cause") for t in scored if t["score"].get("cause")),
    )


def write_summary(results: dict, path: Path):
    m, ts = results["meta"], results["trials"]
    head = ("| {} | n | correct | false neg | false pos | survey ok | judge consistent |",
            "|---|---|---|---|---|---|---|")

    def row(label, st):
        return (f"| {label} | {st['n']} | {_fmt(st['acc'])} | {st['fn']}/{st['n_pos']} | "
                f"{st['fp']}/{st['n_neg']} | {_fmt(st['survey'])} | {_fmt(st['judge'])} |")

    lines = ["# VLM final-evaluation test", "",
             f"- scenes: {len(results['scenes'])} from `{m['scenes_file']}`",
             f"- model: `{m['model']}` (temp {m['temperature']}), "
             f"{'two-pass survey -> judge' if m['two_pass'] else 'single-call'}"
             f"{' — survey sees BEFORE (round-1 decompose frame) + AFTER' if m.get('initial') else ''}"
             f"{' — object inventory from BEFORE (monitor grounder), judge sees appearance' if m.get('inventory') else ''}"
             f"{' — strict judge (order ignored, exact set written out first, names may differ)' if m.get('strict_judge') else ''}"
             f"{' — sets judge (requirements from instruction + names only, compared in code)' if m.get('sets_judge') else ''}"
             f"{' — survey reports looks per object, passed to the requirements call' if m.get('looks') else ''}"
             f"{' — requirements call also sees the BEFORE frame' if m.get('sets_image') else ''}"
             + (" — orchestrator evaluate_final (ported BEFORE/AFTER path)" if m.get("orchestrator") else "")
             + (f" — survey reports tray / tray edge / table; tray edge counts as "
                f"{edge_to(m['tray_policy'])} ({m['tray_policy']} policy)" if m.get("tray_policy") else ""),
             f"- trials per prompt per scene: {m['trials']}"
             + (f" — fixed subset of {m['sample']} cases (seed {m['seed']})" if m.get("sample") else ""),
             f"- unscored cases (hinge on a partial object / name an absent one): "
             f"{sum(t['score']['expected'] is None for t in ts)}", "",
             "false neg = done but judged incomplete (needless retry round); "
             "false pos = not done but judged complete (task stops early).", "",
             "## Per category", "", head[0].format("category"), head[1]]
    for cat in CATEGORIES:
        cts = [t for t in ts if t["category"] == cat]
        if cts:
            lines.append(row(cat, group_stats(cts)))
    lines.append(row("**all**", group_stats(ts)))

    lines += ["", "## Per prompt", "", head[0].format("category | prompt"),
              "|---" + head[1]]
    for key, spec in results["prompts"].items():
        pts = [t for t in ts if t["key"] == key]
        if pts:
            lines.append(row(f"{spec['category']} | {spec['prompt']}", group_stats(pts)))

    lines += ["", "## Per scene", "",
              "| scene | banana | toy | pouch | " + " | ".join(CATEGORIES) + " | all | survey ok | notes |",
              "|" + "---|" * (len(CATEGORIES) + 7)]
    for sc in results["scenes"]:
        sts = [t for t in ts if t["scene"] == sc["name"]]
        if not sts:
            continue
        per_cat = [group_stats([t for t in sts if t["category"] == c])["acc"] for c in CATEGORIES]
        st = group_stats(sts)
        loc = sc["locations"]
        lines.append(f"| {sc['name']} | {loc['banana']} | {loc['toy']} | {loc['pouch']} | "
                     + " | ".join(_fmt(v) for v in per_cat)
                     + f" | {_fmt(st['acc'])} | {_fmt(st['survey'])} | {sc.get('notes', '')} |")

    overall = group_stats(ts)
    n_wrong = sum(overall["causes"].values())
    if n_wrong:
        lines += ["", "## Error attribution", "", "| cause | cases | share | meaning |",
                  "|---|---|---|---|"]
        for key, meaning in CAUSES.items():
            if overall["causes"][key]:
                lines.append(f"| {key} | {overall['causes'][key]} | "
                             f"{overall['causes'][key] / n_wrong:.0%} | {meaning} |")
        wrong_objs = collections.Counter(o for t in ts if t["score"].get("cause") == "perception"
                                         for o in t["score"]["survey_wrong"])
        if wrong_objs:
            lines += ["", "Perception errors by object: "
                      + ", ".join(f"{o} {n}" for o, n in wrong_objs.most_common())]

    fixture = [t for t in ts if t["score"].get("survey_fixture_rows")]
    retry = _rate([t for t in ts if t["success"] is False], "retry_ok")
    lines += ["", "## Survey side effects", "",
              f"- surveys listing a fixture (tray / arm / table / hand) as an object: "
              f"{len(fixture)}/{sum(bool(t['surveyed']) for t in ts)}",
              f"- retry targets right (remaining_from_survey == objects still on the table), "
              f"over cases judged incomplete: {_fmt(retry[0])} (n={retry[1]})"]
    path.write_text("\n".join(lines) + "\n")


def plot(results: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Same ink and categorical order as vlm_decompose_test's figure.
    TEXT, TEXT2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    plt.rcParams.update({"font.size": 9, "text.color": TEXT, "axes.labelcolor": TEXT2,
                         "xtick.color": TEXT2, "ytick.color": TEXT})
    ts = results["trials"]
    cats = [c for c in CATEGORIES if any(t["category"] == c for t in ts)]
    stats = [group_stats([t for t in ts if t["category"] == c]) for c in cats]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 0.5 * len(cats) + 2), facecolor=BG,
                                 gridspec_kw=dict(wspace=0.45))
    y = range(len(cats))
    acc = [s["acc"] or 0 for s in stats]
    ax.barh(y, acc, height=0.55, color=SERIES[0])
    for yi, s in zip(y, stats):
        ax.text((s["acc"] or 0) + 0.02, yi,
                f"{_fmt(s['acc'])}  FN {s['fn']}/{s['n_pos']} · FP {s['fp']}/{s['n_neg']}",
                va="center", fontsize=8.5, color=TEXT)
    ax.set_xlim(0, 1.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Final verdict correct, by preference category", loc="left", fontsize=10)

    left = [0] * len(cats)
    for ci, cause in enumerate(CAUSES):  # colour stays tied to the cause, even if one is absent
        vals = [s["causes"][cause] for s in stats]
        if not any(vals):
            continue
        bx.barh(y, vals, left=left, height=0.55, color=SERIES[ci + 1], label=cause,
                edgecolor=BG, linewidth=2)
        left = [a + b for a, b in zip(left, vals)]
    bx.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=len(CAUSES),
              fontsize=8)
    bx.set_title("Wrong verdicts, by cause", loc="left", fontsize=10)
    bx.set_xlabel("cases")

    for a in (ax, bx):
        a.set_yticks(list(y), cats)
        a.invert_yaxis()
        a.grid(axis="x", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for s in ["top", "right", "left"]:
            a.spines[s].set_visible(False)
        a.spines["bottom"].set_color(GRID)
        a.tick_params(length=0)

    m = results["meta"]
    fig.suptitle(f"Final-evaluation judge on {len(results['scenes'])} labelled final scenes",
                 x=0.01, ha="left", fontsize=12)
    fig.text(0.01, -0.07, f"{m['model']}, temp {m['temperature']}, {m['trials']} trial(s)/prompt/scene · "
             "FN = done but judged incomplete · FP = not done but judged complete",
             fontsize=7.5, color=TEXT2)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def report(results_path: Path, scenes_file: str | None = None):
    results = json.loads(results_path.read_text())
    # Re-score against the current labels so --replot picks up label fixes.
    labels = {s["name"]: s for s in json.loads(
        Path(scenes_file or results["meta"]["scenes_file"]).read_text())["scenes"]}
    results["scenes"] = [{**s, **labels.get(s["name"], {})} for s in results["scenes"]]
    by_name = {s["name"]: s for s in results["scenes"]}
    for t in results["trials"]:
        t["score"] = score_case(results["prompts"][t["key"]], by_name[t["scene"]],
                                t["surveyed"], t["success"], t["remaining"],
                                results["meta"].get("tray_policy"))
    results_path.write_text(json.dumps(results, indent=2))
    write_summary(results, results_path.parent / "summary.md")
    plot(results, results_path.parent / "final_eval_results.png")
    print(f"Wrote {results_path.parent / 'summary.md'}, final_eval_results.png")


def main():
    ap = argparse.ArgumentParser(description="High-level (final) task evaluation tester")
    ap.add_argument("--scenes-file", default=DEFAULT_SCENES, help="Labelled final-scene frames")
    ap.add_argument("--scenes", nargs="+", default=None, help="Only these scene names")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temp", type=float, default=0.1, help="Orchestrator default vlm_temperature")
    ap.add_argument("--trials", type=int, default=1, help="Repeats per prompt per scene")
    ap.add_argument("--categories", default=None, help=f"Comma list from {CATEGORIES}")
    ap.add_argument("--single-call", action="store_true",
                    help="Evaluate with the single-call prompt instead of survey -> judge")
    ap.add_argument("--initial", action="store_true",
                    help="Prototype: the survey also sees the task's initial (round-1 decompose) "
                         "frame and reports each object's before/after support")
    ap.add_argument("--inventory", action="store_true",
                    help="With --initial: build the monitor's scene inventory from the BEFORE frame, "
                         "survey exactly those objects, and give the judge each one's appearance")
    ap.add_argument("--strict-judge", action="store_true",
                    help="With --initial / --rejudge: judge ignores order, writes the required "
                         "tray set before its verdict, and allows differently named objects")
    ap.add_argument("--sets-judge", action="store_true",
                    help="With --initial / --rejudge: a text-only call turns the instruction + object "
                         "names (no locations) into required sets; code compares them with the scene")
    ap.add_argument("--looks", action="store_true",
                    help="With --initial: the survey also reports a location-free 'looks' per object "
                         "(colour, shape, material); the sets judge's requirements call gets it")
    ap.add_argument("--sets-image", action="store_true",
                    help="With --sets-judge: the requirements call also sees the BEFORE frame "
                         "(what each object is), never the final scene")
    ap.add_argument("--orchestrator", action="store_true",
                    help="With --initial: call the orchestrator's evaluate_final(prompt, final, "
                         "initial) — the ported BEFORE/AFTER path — instead of the test prototypes")
    ap.add_argument("--tray-edge", action="store_true",
                    help="With --initial: the survey answers tray / tray edge / table; code maps "
                         "'tray edge' by --tray-policy (partial labels are scored the same way)")
    ap.add_argument("--tray-policy", choices=("loose", "strict"), default="loose",
                    help="With --tray-edge: an object partly on the tray counts as on it (loose) or not")
    ap.add_argument("--sample", type=int, default=None,
                    help="Run a fixed, balanced subset of ~N scene x prompt cases (for manual review)")
    ap.add_argument("--seed", type=int, default=0, help="Seed for --sample")
    ap.add_argument("--keep-unscored", action="store_true",
                    help="Also run cases whose expected verdict is ambiguous")
    ap.add_argument("--out-root", default="vlm_final_eval_test")
    ap.add_argument("--replot", default=None, help="Rescore + rebuild summary from a results.json")
    ap.add_argument("--rejudge", default=None,
                    help="Re-run only the judge on an --initial run's saved surveys (new output folder)")
    args = ap.parse_args()
    if args.strict_judge and not (args.initial or args.rejudge):
        ap.error("--strict-judge applies to the --initial judge (run with --initial or --rejudge)")
    if args.sets_judge and not (args.initial or args.rejudge):
        ap.error("--sets-judge applies to the --initial judge (run with --initial or --rejudge)")
    if args.sets_judge and (args.strict_judge or args.inventory):
        ap.error("--sets-judge replaces the judge; don't combine it with --strict-judge / --inventory")
    if args.orchestrator and (not args.initial or args.sets_judge or args.tray_edge
                              or args.looks or args.inventory or args.strict_judge):
        ap.error("--orchestrator needs --initial and replaces the prototype judge flags")
    if args.tray_edge and not args.initial:
        ap.error("--tray-edge needs --initial (it is a variant of the BEFORE/AFTER survey)")
    if args.sets_image and not args.sets_judge:
        ap.error("--sets-image needs --sets-judge")
    if args.looks and (not args.initial or args.inventory):
        ap.error("--looks needs --initial and replaces --inventory's appearance list")
    if args.inventory and not args.initial:
        ap.error("--inventory needs --initial (the inventory comes from the BEFORE frame)")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    results_path = (Path(args.replot) if args.replot
                    else rejudge(args) if args.rejudge else run(args))
    report(results_path, args.scenes_file if args.replot else None)


if __name__ == "__main__":
    main()
