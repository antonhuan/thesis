"""
Decomposition tester for the orchestrator's two-pass planner.

Feeds preference variants of "put everything on the tray" to the SAME decompose
path the robot uses -- vlm_robot_orchestrator.VLMPlanner.decompose, i.e.

    pass 1  identify_objects         image -> visible_objects
    pass 2  decompose_from_objects   image + visible_objects + instruction -> plan
    (retry) with --reask only: the UNCOVERED_OBJECTS_CONTEXT re-ask when an allowed
            object has no sub-task (off by default, as in the planner)

-- against one or more initial scenes where the pouch, banana and toy are all
visible on the table. Every prompt runs --trials times on every scene. No decompose logic is duplicated here: the planner is called as-is and
vlm.generate is wrapped only to record every raw pass output.

Preference variants (object set X = {banana, toy, pouch}, one tray):
    base          put everything on the tray                     -> all three, any order
    omission      leave one or two objects                       -> X minus excluded
    sequencing    placement order constraints                    -> all three, order checked
    conditional   attribute / if-then selection over the scene   -> the objects satisfying it
    quantity      a number of objects (optionally of a class)    -> n objects from the class

Each trial is auto-labelled (the checks below) and carries an empty
"manual_label" block. Fill manual_label.correct (true/false) -- and optionally
failure_mode / notes -- in results.json, then re-run with --replot: a manual
label overrides the auto label wherever it is set.

Auto checks per trial:
    identified   pass 1 found banana, toy and pouch (synonyms mapped)
    object_set   the planned objects are the expected set (quantity: n of the class)
    order        every required (before, after) pair is respected
    count        no object planned twice, and the plan has the expected length
    destination  every sub-task puts its object on the tray
    correct      all of the above

Usage (inside the lerobot-so101 container, cwd /workspace):
    python vlm_decompose_test.py                              # default scene, 3 trials/prompt
    python vlm_decompose_test.py --image runs/.../decompose_X.png --trials 5
    python vlm_decompose_test.py --image a.png b.png c.png        # several initial scenes
    python vlm_decompose_test.py --scene-dir decompose_scenes/     # every .png/.jpg in a folder
    python vlm_decompose_test.py --categories omission,quantity
    python vlm_decompose_test.py --replot vlm_decompose_test/<stamp>/results.json
    python vlm_decompose_test.py --sample 200                     # fixed ~200-pair subset
    python vlm_decompose_test.py --replot <stamp>/results.json --sample 200   # subset an existing run

Outputs (vlm_decompose_test/<stamp>/):
    results.json     per-trial log: raw pass outputs, parsed plan, auto + manual labels
    summary.md       per-prompt table
    decompose_results.png
    scenes/<name>.png  copy of each input frame (name = its run dir, else file stem)
"""

import argparse
import collections
import json
import logging
import random
import re
import shutil
import time
from collections import OrderedDict
from datetime import datetime
from itertools import product
from pathlib import Path

from PIL import Image

OBJECTS = ["banana", "toy", "pouch"]
CATEGORIES = ["base", "omission", "sequencing", "conditional", "quantity"]
CHECKS = ["identified", "object_set", "order", "count", "destination"]

DEFAULT_IMAGE = "runs/run_2026-09-17_07-24-42/01/decompose_20260917_072520.png"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"  # what the orchestrator is run with

# Words the planner may use for each object (identification names are free-form).
SYNONYMS = {
    "banana": ["banana", "fruit"],
    "toy": ["toy", "plush", "moose", "stuffed", "animal", "teddy", "deer", "reindeer", "doll"],
    "pouch": ["pouch", "pencil case", "case", "bag", "wallet", "purse", "cylinder", "cylindrical", "roll"],
}


# --------------------------------------------------------------------------- #
# Prompt set. `expected` is the exact object set to place; quantity prompts
# instead give `pool` + `n` (any n distinct objects from the pool is correct).
# `order` lists (before, after) pairs the plan must respect.
# --------------------------------------------------------------------------- #
def _p(category, prompt, expected=None, order=(), pool=None, n=None):
    return dict(category=category, prompt=prompt,
                expected=sorted(expected) if expected is not None else None,
                order=[list(o) for o in order], pool=pool, n=n)


PROMPTS = OrderedDict([
    # --- Base -------------------------------------------------------------
    ("base",                 _p("base", "put everything on the tray", OBJECTS)),
    # --- Omission ---------------------------------------------------------
    ("omit banana",          _p("omission", "put everything on the tray except the banana", ["toy", "pouch"])),
    ("omit toy",             _p("omission", "put everything on the tray except the toy", ["banana", "pouch"])),
    ("omit pouch",           _p("omission", "put everything on the tray except the pouch", ["banana", "toy"])),
    ("omit banana+toy",      _p("omission", "put everything on the tray except the banana and the toy", ["pouch"])),
    ("omit toy+pouch",       _p("omission", "put everything on the tray but leave the toy and the pouch", ["banana"])),
    ("omit don't touch",     _p("omission", "put everything on the tray, don't touch the pouch", ["banana", "toy"])),
    # --- Sequencing -------------------------------------------------------
    ("seq banana>toy>pouch", _p("sequencing", "put the banana on the tray first, then the toy, then the pouch",
                                OBJECTS, [("banana", "toy"), ("toy", "pouch")])),
    ("seq pouch>banana>toy", _p("sequencing", "put the pouch on the tray first, then the banana, then the toy",
                                OBJECTS, [("pouch", "banana"), ("banana", "toy")])),
    ("seq toy first",        _p("sequencing", "put everything on the tray, starting with the toy",
                                OBJECTS, [("toy", "banana"), ("toy", "pouch")])),
    ("seq pouch last",       _p("sequencing", "put everything on the tray, but the pouch goes last",
                                OBJECTS, [("banana", "pouch"), ("toy", "pouch")])),
    ("seq banana after toy", _p("sequencing", "put everything on the tray, put the banana after the toy",
                                OBJECTS, [("toy", "banana")])),
    # --- Conditional ------------------------------------------------------
    ("cond fruit",           _p("conditional", "put the fruit on the tray", ["banana"])),
    ("cond brown",           _p("conditional", "put the brown objects on the tray", ["toy", "pouch"])),
    ("cond not edible",      _p("conditional", "put everything that is not edible on the tray", ["toy", "pouch"])),
    ("cond soft",            _p("conditional", "put only the soft, furry things on the tray", ["toy"])),
    ("cond if banana",       _p("conditional", "if there is a banana on the table, put only the banana on the tray; "
                                               "otherwise put everything on the tray", ["banana"])),
    ("cond if apple",        _p("conditional", "if there is an apple on the table, put the apple on the tray; "
                                               "otherwise put the toy on the tray", ["toy"])),
    # --- Quantity ---------------------------------------------------------
    ("qty one",              _p("quantity", "put one object on the tray", pool=OBJECTS, n=1)),
    ("qty two",              _p("quantity", "put two objects on the tray", pool=OBJECTS, n=2)),
    ("qty three",            _p("quantity", "put all three objects on the tray", pool=OBJECTS, n=3)),
    ("qty one brown",        _p("quantity", "put one of the brown objects on the tray", pool=["toy", "pouch"], n=1)),
    ("qty two not banana",   _p("quantity", "put two objects on the tray, but not the banana", pool=["toy", "pouch"], n=2)),
])


# --------------------------------------------------------------------------- #
# Auto labelling
# --------------------------------------------------------------------------- #
def canonical(text: str) -> str | None:
    """Map a free-form object name to banana/toy/pouch (first match by position)."""
    t = text.lower()
    best, best_pos = None, len(t) + 1
    for obj, words in SYNONYMS.items():
        for w in words:
            m = re.search(rf"\b{re.escape(w)}", t)
            if m and m.start() < best_pos:
                best, best_pos = obj, m.start()
    return best


# Scene fixtures the identify pass lists alongside the objects; never a thing to move.
FIXTURES = ["tray", "table", "robot", "arm", "gripper", "wire", "cable", "label", "surface",
            "camera", "cord"]


def elimination_map(identified: list[str]) -> dict[str, str]:
    """Resolve vague identify-pass names ("brown object", "object") by elimination.

    Drops fixtures and names that already map to an object; if exactly one of
    banana/toy/pouch is then unaccounted for and exactly one unknown name is
    left, that name is that object. Otherwise nothing is resolved.
    """
    known = {canonical(o) for o in identified} - {None}
    unknown = [o for o in identified
               if canonical(o) is None and not any(re.search(rf"\b{f}", o.lower()) for f in FIXTURES)]
    missing = [o for o in OBJECTS if o not in known]
    if len(missing) == 1 and len(unknown) == 1:
        return {unknown[0].lower(): missing[0]}
    return {}


def subtask_object(subtask: str, resolved: dict[str, str] | None = None) -> str | None:
    """The object being moved: the first object mentioned before the destination."""
    head = re.split(r"\b(on|onto|in|into|to|next to)\s+the\s+tray", subtask.lower())[0]
    obj = canonical(head) or canonical(subtask)
    if obj is None and resolved:
        obj = next((o for name, o in resolved.items() if name in head), None)
    return obj


def auto_label(spec: dict, identified: list[str], subtasks: list[str] | None,
               error: str | None) -> dict:
    labels = {c: False for c in CHECKS}
    resolved = elimination_map(identified)
    labels["identified"] = set(OBJECTS) <= ({canonical(o) for o in identified} | set(resolved.values()))
    planned = [subtask_object(s, resolved) for s in (subtasks or [])]
    extra = dict(planned=planned, resolved_by_elimination=resolved)
    if error is not None or subtasks is None:
        return dict(**labels, correct=False, failure_mode="parse_error", **extra)

    uniq = [p for p in planned if p]
    if spec["n"] is not None:
        labels["object_set"] = set(uniq) <= set(spec["pool"]) and len(set(uniq)) == spec["n"]
        expected_len = spec["n"]
    else:
        labels["object_set"] = set(uniq) == set(spec["expected"])
        expected_len = len(spec["expected"])
    labels["count"] = len(uniq) == len(set(uniq)) == len(planned) == expected_len
    labels["order"] = all(
        a in uniq and b in uniq and uniq.index(a) < uniq.index(b) for a, b in spec["order"])
    labels["destination"] = bool(subtasks) and all("tray" in s.lower() for s in subtasks)
    correct = all(labels.values())

    failure = None
    moved_fixture = any(o is None and any(re.search(rf"\b{f}", st.lower()) for f in FIXTURES
                                          if f != "tray")
                        for o, st in zip(planned, subtasks))
    if moved_fixture:
        failure = "planned_fixture"  # e.g. "put the robot arm on the tray"
    elif not correct:
        for check, mode in [("identified", "missed_object"), ("object_set", "wrong_objects"),
                            ("count", "wrong_count"), ("order", "wrong_order"),
                            ("destination", "wrong_destination")]:
            if not labels[check]:
                failure = mode
                break
    return dict(**labels, correct=correct, failure_mode=failure, **extra)


def first_pass_subtasks(trial: dict) -> list[str] | None:
    """Sub-tasks of the first decompose pass, i.e. before the uncovered-object re-ask."""
    first = next((p for p in trial["passes"] if p["pass_name"] == "decompose"), None)
    if first is None:
        return None
    try:
        parsed = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", first["output"].strip()))
        return parsed["subtasks"] if isinstance(parsed, dict) else parsed
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def first_pass_correct(spec: dict, trial: dict) -> bool:
    """Would the plan have been correct WITHOUT the uncovered-object re-ask?

    Scores the subtasks of the first decompose pass; equals the final label when
    no retry happened.
    """
    subtasks = first_pass_subtasks(trial)
    return subtasks is not None and auto_label(spec, trial["identified"], subtasks, None)["correct"]


# --------------------------------------------------------------------------- #
# Failure attribution: perception vs preference reasoning
# --------------------------------------------------------------------------- #
# A failed plan is re-scored against the preference RESTRICTED to the objects the
# identify pass actually perceived. If it is correct there, the preference
# reasoning was right and the failure is perception's.
CAUSES = OrderedDict([
    ("perception_missing",       ("perception", "missed a needed object; reasoning correct on what it saw")),
    ("perception_extra",         ("perception", "planned a non-object (arm, wire, ...); reasoning otherwise correct")),
    ("perception_missing_extra", ("perception", "both of the above; reasoning correct on what it saw")),
    ("grounding",                ("grounding",  "object perceived but named vaguely, so the preference could not refer to it")),
    ("reasoning_reask",          ("reasoning",  "full perception; first plan right, the uncovered-object re-ask broke it")),
    ("reasoning",                ("reasoning",  "full perception; wrong plan")),
    ("mixed",                    ("mixed",      "perception error AND wrong plan on the objects it saw")),
    ("parse_error",              ("other",      "planner output could not be parsed")),
    ("not_a_failure",            ("none",       "missed object was not needed; the plan is correct")),
])


def plan_ok_on_perceived(spec: dict, planned: list, subtasks: list[str], perceived: set) -> bool:
    """Plan correct against the preference restricted to the perceived objects."""
    objs = [o for o in planned if o]
    if len(objs) != len(set(objs)) or not all("tray" in st.lower() for st in subtasks):
        return False
    if spec["n"] is not None:
        pool = [o for o in spec["pool"] if o in perceived]
        return set(objs) <= set(pool) and len(objs) == min(spec["n"], len(pool))
    exp = {o for o in spec["expected"] if o in perceived}
    order = [(a, b) for a, b in spec["order"] if a in perceived and b in perceived]
    return set(objs) == exp and all(objs.index(a) < objs.index(b) for a, b in order)


def failure_cause(spec: dict, trial: dict) -> str | None:
    """Why an auto-labelled failure failed (None for a correct trial). See CAUSES."""
    if trial["auto_label"]["correct"]:
        return None
    if trial["subtasks"] is None:
        return "parse_error"
    resolved = elimination_map(trial["identified"])
    perceived = ({canonical(o) for o in trial["identified"]} | set(resolved.values())) & set(OBJECTS)
    missed = set(OBJECTS) - perceived
    needed = set(spec["pool"] if spec["n"] is not None else spec["expected"])
    planned = [subtask_object(st, resolved) for st in trial["subtasks"]]
    extra = any(o is None for o in planned)
    if plan_ok_on_perceived(spec, planned, trial["subtasks"], perceived):
        if missed & needed:
            return "perception_missing_extra" if extra else "perception_missing"
        if extra:
            return "perception_extra"
        return "not_a_failure"
    # Wrong on what it saw. A wrongly-planned object the model only named vaguely
    # ("brown object" for the excluded pouch) is a grounding failure, not reasoning.
    if any(o in set(resolved.values()) and o not in needed for o in planned):
        return "grounding"
    if missed or extra:
        return "mixed"
    first = first_pass_subtasks(trial)
    if trial["retried"] and first is not None and plan_ok_on_perceived(
            spec, [subtask_object(st, resolved) for st in first], first, perceived):
        return "reasoning_reask"
    return "reasoning"


def effective_correct(trial: dict) -> tuple[bool, bool]:
    """(correct, is_manual): the manual label wins when it is filled in."""
    manual = trial.get("manual_label", {}).get("correct")
    if manual is not None:
        return bool(manual), True
    return bool(trial["auto_label"]["correct"]), False


def scene_name(path: Path, taken: set) -> str:
    """Short unique label: the run_<stamp> dir the frame came from, else its stem."""
    run_dir = next((p.name for p in path.parents if p.name.startswith("run_")), None)
    base = run_dir[len("run_"):] if run_dir else path.stem
    name, i = base, 2
    while name in taken:
        name, i = f"{base}_{i}", i + 1
    taken.add(name)
    return name


def scene_paths(args) -> list[Path]:
    paths = [Path(p) for p in args.image] if args.image else []
    if args.scene_dir:
        paths += sorted(p for p in Path(args.scene_dir).iterdir()
                        if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    return paths or [Path(DEFAULT_IMAGE)]


# --------------------------------------------------------------------------- #
# Running the planner
# --------------------------------------------------------------------------- #
def _split(n: int, k: int) -> list[int]:
    """n into k near-equal shares, larger shares first."""
    return [n // k + (i < n % k) for i in range(k)]


def sample_pairs(scenes: list[str], keys: list[str], n: int, seed: int = 0,
                 per_category: int | None = None) -> list[tuple[str, str]]:
    """A fixed, reviewable subset of (scene, prompt) pairs.

    Shares: with `per_category`, every preference category gets that many pairs, split
    near-equally over its prompts (base, one prompt, takes all its pairs from one
    prompt); otherwise every prompt gets an equal share of n. Within a prompt each pick
    goes to the least-used scene so far (ties: seeded shuffle) and no scene repeats, so
    the subset spreads evenly over the scenes. Same scenes + prompts + seed -> same subset.
    """
    if per_category:
        quota = {}
        for cat in CATEGORIES:
            ks = [k for k in keys if PROMPTS[k]["category"] == cat]
            quota.update(zip(ks, _split(per_category, len(ks)) if ks else []))
    else:
        quota = dict(zip(keys, _split(n, len(keys))))
    rng = random.Random(seed)
    use = collections.Counter()
    pairs = []
    for key in keys:
        cand = scenes[:]
        rng.shuffle(cand)
        for _ in range(min(quota[key], len(cand))):
            pick = min(cand, key=lambda sc: use[sc])  # ties: shuffled order
            cand.remove(pick)
            use[pick] += 1
            pairs.append((pick, key))
    return pairs


def subset_results(results_path: Path, n: int | None, seed: int,
                   per_category: int | None = None) -> Path:
    """Copy an existing run's trials for the --sample / --per-category subset into
    <run>_sample<n>/ or <run>_cat<n>/."""
    results = json.loads(results_path.read_text())
    normalise(results)
    ran = [sc["name"] for sc in results["meta"]["scenes"]
           if any(t["scene"] == sc["name"] for t in results["trials"])]
    keep = set(sample_pairs(ran, list(results["prompts"]), n, seed, per_category))
    results["trials"] = [t for t in results["trials"] if (t["scene"], t["key"]) in keep]
    results["meta"].update(sample=n, per_category=per_category, seed=seed,
                           subset_of=str(results_path))
    tag = f"cat{per_category}" if per_category else f"sample{n}"
    out = results_path.parent.with_name(f"{results_path.parent.name}_{tag}") / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    return out


# --one-pass: the planner as it was before the two-pass split (git b0c32a51^, vlm.py,
# 2026-08-04 — the commit after introduced identify -> decompose). One call does
# everything: list the visible objects, the excluded and allowed ones, and the
# sub-tasks, from the image. Only this prompt differs from today's single-call
# fallback (one rule line about the robot arm); the call itself — "[top-down camera]"
# + image + "Instruction: ...", same sampling, same JSON — is VLMPlanner.decompose with
# two_pass=False.
ONE_PASS_DECOMPOSITION_PROMPT = """You are a task planner for an orange tabletop robot arm (SO-101, 6-DOF). You receive a natural language instruction and a camera observation.

You MUST respond in the following JSON format exactly:

{
  "visible_objects": ["list", "of", "objects", "you", "see"],
  "excluded_objects": ["objects", "the", "instruction", "says", "to", "leave"],
  "allowed_objects": ["visible", "minus", "excluded"],
  "subtasks": ["put the X on the tray", "put the Y next to the X"]
}

Rules:
- The orange robot arm/kumquat is not an object. Do not include it in visible_objects.
- excluded_objects: any object the instruction says to leave, skip, ignore, or not touch. If none, use [].
- allowed_objects: every object in visible_objects that is NOT in excluded_objects.
- subtasks: one sub-task per allowed object ONLY. Each sub-task is a single pick-and-place action.
- Each sub-task must specify both the object AND its destination (e.g. "put the fork next to the plate", "put the bowl on the tray", "put the plate on the placemat").
- Infer the destination for each object from the instruction and common sense. If the instruction gives a specific destination, use it. If the instruction implies an arrangement (e.g. "set the table"), use spatial language appropriate to the task (e.g. "next to", "on", "in front of").
- Destinations MUST be grounded in the scene. Only use objects or surfaces you can see in the camera observation as destinations.
- If the instruction is vague about destination (e.g. "away", "clean up", "tidy up"), choose the most reasonable visible container or surface (e.g. a tray, bin, or box) as the destination.
- Never use a destination that is not visible in the scene.
- If the instruction uses a category word (e.g. "food", "drinks", "utensils"), identify which visible objects belong to that category and treat them all as excluded (or included).
- EVERY object in allowed_objects MUST have exactly one sub-task. If allowed_objects is not empty, subtasks CANNOT be empty.
- Order subtasks logically. Place base objects before objects that go on top of or relative to them (e.g. place the plate before placing the fork next to the plate).
- visible_objects should include ALL objects and surfaces you see, including containers and destinations (trays, bowls, boxes, plates). An object can be both a destination for one subtask and a moved object in another.

Examples:

Instruction: "put everything on the tray but leave the banana"
{"visible_objects": ["apple", "banana", "orange"], "excluded_objects": ["banana"], "allowed_objects": ["apple", "orange"], "subtasks": ["put the apple on the tray", "put the orange on the tray"]}

Instruction: "clean up the table, don't touch the cup"
{"visible_objects": ["cup", "plate", "fork"], "excluded_objects": ["cup"], "allowed_objects": ["plate", "fork"], "subtasks": ["put the plate on the tray", "put the fork on the tray"]}

Instruction: "put the apple on the tray"
{"visible_objects": ["apple", "banana"], "excluded_objects": [], "allowed_objects": ["apple"], "subtasks": ["put the apple on the tray"]}

Instruction: "put everything away but leave the food"
{"visible_objects": ["banana", "apple", "cup", "book"], "excluded_objects": ["banana", "apple"], "allowed_objects": ["cup", "book"], "subtasks": ["put the cup on the tray", "put the book on the tray"]}

Instruction: "stack the blocks"
{"visible_objects": ["red block", "blue block", "green block"], "excluded_objects": [], "allowed_objects": ["red block", "blue block", "green block"], "subtasks": ["put the blue block on the red block", "put the green block on the blue block"]}
"""

# --robot-parts-rule: one planning-time line (no code-side filtering). The decompose
# prompt makes its object lists partition everything it is given, so a robot part the
# identify pass listed ("orange arm", "wires") had nowhere to go but allowed_objects.
ROBOT_PARTS_RULE = ("- The visible_objects list comes from a separate perception step and can include "
                    "parts of the robot itself — its arm, gripper, wires or label. Those are never "
                    "moved: put them in excluded_objects and give them no sub-task.\n")
_RULE_ANCHOR = "- subtasks: one sub-task per allowed object ONLY."


def with_robot_parts_rule(prompt: str) -> str:
    assert prompt.count(_RULE_ANCHOR) == 1, "decompose prompt layout changed"
    return prompt.replace(_RULE_ANCHOR, ROBOT_PARTS_RULE + _RULE_ANCHOR)


def with_ignored_group(prompt: str) -> str:
    """--ignored-group: a fifth group, ignored_objects, for listed items that are not
    movable objects. The prompt otherwise forces every listed item into a group, so a
    distractor the identify pass reported (equipment, background) ends up planned."""
    def sub(old, new):
        nonlocal prompt
        assert prompt.count(old) == 1, f"decompose prompt layout changed: {old[:40]}"
        prompt = prompt.replace(old, new)
    sub('"allowed_objects": ["objects", "to", "move"]',
        '"ignored_objects": ["listed", "items", "that", "are", "not", "objects"],\n'
        '  "allowed_objects": ["objects", "to", "move"]')
    sub("The four object lists PARTITION", "The five object lists PARTITION")
    sub("exactly one of excluded_objects, completed_objects, allowed_objects, or the destinations.",
        "exactly one of excluded_objects, completed_objects, ignored_objects, allowed_objects, or the "
        "destinations.")
    sub("- allowed_objects: the objects that still need to be moved",
        "- ignored_objects: listed items that are not movable objects for this task — equipment, "
        "parts of the robot, cables, devices, or anything in the background beyond the workspace. "
        "The list can contain such items because it comes from a separate perception step. Use the "
        "camera image to judge. Never plan a sub-task for them. If none, use [].\n"
        "- allowed_objects: the objects that still need to be moved")
    sub("MINUS completed, MINUS destinations.", "MINUS completed, MINUS ignored, MINUS destinations.")
    sub("A retry pass, where",
        'A listed item that is not a movable object goes in ignored_objects, even when the instruction '
        'says "everything":\n'
        'Visible objects: ["apple", "keyboard", "cup", "monitor", "tray"]\n'
        'Instruction: "put everything on the tray"\n'
        '{"visible_objects": ["apple", "keyboard", "cup", "monitor", "tray"], "excluded_objects": [], '
        '"completed_objects": [], "ignored_objects": ["keyboard", "monitor"], "allowed_objects": '
        '["apple", "cup"], "subtasks": ["put the apple on the tray", "put the cup on the tray"]}\n\n'
        "A retry pass, where")
    return prompt


# --role-identify: the identify pass labels each item's role; only "object" items reach the
# planner (with "surface" items as destinations). The model decides what is a distractor,
# instead of a word list.
ROLES = ("object", "robot", "surface", "background")
_EXAMPLE_ROLE = {"tray": "surface", "table": "surface", "computer": "background"}


def with_roles(prompt: str) -> str:
    def sub(old, new):
        nonlocal prompt
        assert prompt.count(old) == 1, f"identify prompt layout changed: {old[:40]}"
        prompt = prompt.replace(old, new)
    sub('{"visible_objects": [{"name": "object name", "evidence": "brief description of what you see and where"}, ...]}',
        '{"visible_objects": [{"name": "object name", "evidence": "brief description of what you see and where", '
        '"role": "object | robot | surface | background"}, ...]}')
    sub("- Fill in \"evidence\" for every object:",
        "- Give every entry a \"role\": \"object\" for a loose item on the table that could be picked up "
        "and moved; \"robot\" for any part of the robot (its arm, gripper, wires, cables, labels); "
        "\"surface\" for a surface or container things rest on or are placed in (the table, a tray, a "
        "bowl, a box); \"background\" for equipment or anything at the edge of or beyond the "
        "workspace that is not part of the task (equipment, electronics, anything outside the work area). Fill in "
        "\"evidence\" first and decide the role from it.\n"
        "- Fill in \"evidence\" for every object:")
    sub("- The orange robotic gripper arm is part of the robot, NOT an object — do not include it regardless of its color or shape.",
        "- The orange robotic gripper arm is part of the robot, NOT an object — if you list it or its "
        "wires, give them the role \"robot\".")

    def add_role(m):
        name = m.group(1)
        return m.group(0)[:-1] + f', "role": "{_EXAMPLE_ROLE.get(name, "object")}"}}'
    ex_start = prompt.index("Examples:")
    prompt = prompt[:ex_start] + re.sub(r'\{"name": "([^"]+)", "evidence": "[^"]*"\}', add_role,
                                        prompt[ex_start:])
    return prompt


def role_rows(output: str) -> list[dict]:
    text = output.strip()
    for cand in (text, text[text.find("{"): text.rfind("}") + 1]):
        try:
            parsed = json.loads(cand)
            rows = parsed.get("visible_objects") if isinstance(parsed, dict) else parsed
            return [r for r in rows or [] if isinstance(r, dict) and r.get("name")]
        except (json.JSONDecodeError, AttributeError):
            continue
    return []


def install_role_identify(vlm, orch, prompt: str):
    """identify_objects that asks for roles and returns only the "object" items plus the
    "surface" items (destinations); the full role rows are logged with the pass."""
    def identify(model, processor, image_content, temperature=0.7):
        user = list(image_content) + [{"type": "text", "text": "\nList every visible object and surface."}]
        out = vlm.generate(model, processor, [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {"role": "user", "content": user}], temperature=temperature)
        rows = role_rows(out)
        kept = [str(r["name"]) for r in rows if str(r.get("role", "object")).lower() in ("object", "surface")]
        dropped = [(str(r["name"]), r.get("role")) for r in rows if str(r["name"]) not in kept]
        if dropped:
            logging.info(f"Identify roles set aside: {dropped}")
        return kept
    orch.identify_objects = identify


def redecompose(args) -> Path:
    """Re-run only the decompose pass of a two-pass run, on each trial's recorded
    identify output (planner.decompose(..., objects=identified)), optionally with
    --robot-parts-rule. Perception is held fixed, so any change is the planner's."""
    import vlm
    import vlm_robot_orchestrator as orch

    src = Path(args.redecompose)
    results = json.loads(src.read_text())
    normalise(results)
    if results["meta"].get("one_pass"):
        raise SystemExit("--redecompose needs a two-pass run (it reuses the identify pass)")
    passes: list[dict] = []
    real = vlm.generate

    def recording(model, processor, messages, *a, **kw):
        t0 = time.time()
        out = real(model, processor, messages, *a, **kw)
        passes.append(dict(pass_name="decompose", output=out, latency_s=round(time.time() - t0, 2)))
        return out

    vlm.generate = recording
    if args.robot_parts_rule:
        vlm.DECOMPOSITION_FROM_OBJECTS_SYSTEM_PROMPT = with_robot_parts_rule(
            vlm.DECOMPOSITION_FROM_OBJECTS_SYSTEM_PROMPT)
    if args.ignored_group:
        vlm.DECOMPOSITION_FROM_OBJECTS_SYSTEM_PROMPT = with_ignored_group(
            vlm.DECOMPOSITION_FROM_OBJECTS_SYSTEM_PROMPT)
    m = results["meta"]
    planner = orch.VLMPlanner(args.model, temperature=m["temperature"], two_pass=True,
                              reask_uncovered=m.get("reask_uncovered", False),
                              filter_fixtures=not args.no_fixture_filter)
    frames = {sc["name"]: Image.open(sc["path"]).convert("RGB") for sc in m["scenes"]}
    out_dir = Path(args.out_root) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    m.update(redecompose_of=str(src), robot_parts_rule=args.robot_parts_rule,
             ignored_group=args.ignored_group,
             filter_fixtures=not args.no_fixture_filter,
             created=datetime.now().isoformat(timespec="seconds"))
    ts = results["trials"]
    for i, t in enumerate(ts, 1):
        spec = results["prompts"][t["key"]]
        identify = [p for p in t["passes"] if p["pass_name"] == "identify"]
        passes.clear()
        error, subtasks = None, None
        t0 = time.time()
        try:
            # As the orchestrator does after the identify pass (vlm_filter_fixtures).
            objects = (orch.planner_objects(t["identified"])[0] if planner.filter_fixtures
                       else t["identified"])
            subtasks = planner.decompose(spec["prompt"], frames[t["scene"]], objects=objects)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        parsed = {}
        if passes:
            try:
                parsed = json.loads(orch._strip_code_fences(passes[-1]["output"]))
            except json.JSONDecodeError:
                parsed = {}
        parsed = parsed if isinstance(parsed, dict) else {}
        t.update(subtasks=subtasks, error=error, passes=identify + list(passes),
                 visible_objects=parsed.get("visible_objects"),
                 excluded_objects=parsed.get("excluded_objects"),
                 ignored_objects=parsed.get("ignored_objects"),
                 allowed_objects=parsed.get("allowed_objects"),
                 latency_s=round(time.time() - t0, 2),
                 auto_label=auto_label(spec, t["identified"], subtasks, error),
                 manual_label=dict(correct=None, failure_mode=None, notes=""))
        logging.info(f"[{i}/{len(ts)}] {t['scene']} {t['key']}: "
                     f"{'PASS' if t['auto_label']['correct'] else 'FAIL'} planned={t['auto_label']['planned']} "
                     f"{t['auto_label']['failure_mode'] or ''}")
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    vlm.generate = real
    return out_dir / "results.json"


def run(args) -> Path:
    import vlm
    import vlm_robot_orchestrator as orch

    passes: list[dict] = []
    real_generate = vlm.generate

    def recording_generate(model, processor, messages, *a, **kw):
        system = messages[0]["content"][0]["text"]
        name = ("identify" if system in (vlm.IDENTIFICATION_SYSTEM_PROMPT,
                                         with_roles(vlm.IDENTIFICATION_SYSTEM_PROMPT))
                else "decompose" if system == vlm.DECOMPOSITION_FROM_OBJECTS_SYSTEM_PROMPT
                else "one_pass" if system == ONE_PASS_DECOMPOSITION_PROMPT
                else "other")
        user_text = "".join(c.get("text", "") for c in messages[1]["content"] if c.get("type") == "text")
        t0 = time.time()
        out = real_generate(model, processor, messages, *a, **kw)
        passes.append(dict(pass_name=name, user_text=user_text, output=out,
                           latency_s=round(time.time() - t0, 2)))
        return out

    vlm.generate = recording_generate
    real_orch_generate = orch.generate
    real_identify = orch.identify_objects
    if args.ignored_group:
        vlm.DECOMPOSITION_FROM_OBJECTS_SYSTEM_PROMPT = with_ignored_group(
            vlm.DECOMPOSITION_FROM_OBJECTS_SYSTEM_PROMPT)
    if args.role_identify:
        install_role_identify(vlm, orch, with_roles(vlm.IDENTIFICATION_SYSTEM_PROMPT))
    if args.one_pass:
        # The single-call path runs orch.generate with the orchestrator's own prompt
        # constant; swap in the original prompt and record that call too.
        orch.DECOMPOSITION_SYSTEM_PROMPT = ONE_PASS_DECOMPOSITION_PROMPT
        orch.generate = recording_generate

    out_dir = Path(args.out_root) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    (out_dir / "scenes").mkdir(parents=True, exist_ok=True)
    taken: set = set()
    scenes = []
    for path in scene_paths(args):
        name = scene_name(path, taken)
        shutil.copy(path, out_dir / "scenes" / f"{name}{path.suffix}")
        scenes.append(dict(name=name, path=str(path)))

    planner = orch.VLMPlanner(args.model, temperature=args.temp, two_pass=not args.one_pass,
                              filter_fixtures=not args.no_fixture_filter,
                              reask_uncovered=args.reask)
    cats = args.categories.split(",") if args.categories else CATEGORIES
    prompts = OrderedDict((k, v) for k, v in PROMPTS.items() if v["category"] in cats)

    results = dict(
        meta=dict(scenes=scenes, model=args.model, temperature=args.temp, reask_uncovered=args.reask,
                  one_pass=args.one_pass, role_identify=args.role_identify,
                  ignored_group=args.ignored_group,
                  trials=args.trials, created=datetime.now().isoformat(timespec="seconds"),
                  sample=args.sample, per_category=args.per_category, seed=args.seed,
                  scene_objects=OBJECTS,
                  manual_label_help="Set manual_label.correct to true/false (failure_mode/notes "
                                    "optional); re-run with --replot to rebuild the plot."),
        prompts=prompts,
        trials=[],
    )
    keep = (set(sample_pairs([sc["name"] for sc in scenes], list(prompts), args.sample, args.seed,
                             args.per_category))
            if args.sample or args.per_category else None)
    cases = [(scene, (key, spec), t)
             for scene, (key, spec), t in product(scenes, prompts.items(), range(args.trials))
             if keep is None or (scene["name"], key) in keep]
    total = len(cases)
    done = 0
    frames = {sc["name"]: Image.open(sc["path"]).convert("RGB") for sc in scenes}
    for scene, (key, spec), t in cases:
        frame = frames[scene["name"]]
        passes.clear()
        error, subtasks = None, None
        t0 = time.time()
        try:
            subtasks = planner.decompose(spec["prompt"], frame)
        except Exception as e:  # parse failure etc. is a result, not a crash
            error = f"{type(e).__name__}: {e}"
        identified = []
        final_parsed = {}
        for p in passes:
            if p["pass_name"] == "identify":
                identified = vlm.parse_visible_objects(p["output"])
            if p["pass_name"] in ("decompose", "one_pass"):
                try:
                    final_parsed = json.loads(orch._strip_code_fences(p["output"]))
                except json.JSONDecodeError:
                    final_parsed = {}
        if args.one_pass and isinstance(final_parsed, dict):
            # No identify pass: what the one call listed as visible is what it perceived.
            identified = [str(o) for o in final_parsed.get("visible_objects") or []]
        label = auto_label(spec, identified, subtasks, error)
        trial = dict(
            scene=scene["name"], key=key, category=spec["category"], prompt=spec["prompt"], trial=t,
            identified=identified,
            visible_objects=final_parsed.get("visible_objects") if isinstance(final_parsed, dict) else None,
            excluded_objects=final_parsed.get("excluded_objects") if isinstance(final_parsed, dict) else None,
            ignored_objects=final_parsed.get("ignored_objects") if isinstance(final_parsed, dict) else None,
            allowed_objects=final_parsed.get("allowed_objects") if isinstance(final_parsed, dict) else None,
            subtasks=subtasks, error=error,
            retried=sum(p["pass_name"] == "decompose" for p in passes) > 1,
            latency_s=round(time.time() - t0, 2),
            passes=list(passes),
            auto_label=label,
            manual_label=dict(correct=None, failure_mode=None, notes=""),
        )
        results["trials"].append(trial)
        done += 1
        logging.info(f"[{done}/{total}] {scene['name']} {key} #{t}: {'PASS' if label['correct'] else 'FAIL'} "
                     f"planned={label['planned']} {label['failure_mode'] or ''}")
        # Save as we go so a crash keeps what already ran.
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    vlm.generate = real_generate
    orch.generate = real_orch_generate
    orch.identify_objects = real_identify
    return out_dir / "results.json"


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def normalise(results: dict):
    """Older single-scene logs had meta.image and no per-trial scene."""
    m = results["meta"]
    if "scenes" not in m:
        m["scenes"] = [dict(name="scene", path=m.get("image", ""))]
    for t in results["trials"]:
        t.setdefault("scene", m["scenes"][0]["name"])


def scene_category_table(results: dict) -> tuple[list[str], list[str], list[list]]:
    """(scenes, categories, rate[scene][category]) of effective correctness."""
    ran = {t["scene"] for t in results["trials"]}
    scenes = [sc["name"] for sc in results["meta"]["scenes"] if sc["name"] in ran]  # partial runs
    cats = [c for c in CATEGORIES if any(t["category"] == c for t in results["trials"])]
    grid = []
    for sc in scenes:
        row = []
        for c in cats:
            ts = [t for t in results["trials"] if t["scene"] == sc and t["category"] == c]
            row.append(sum(effective_correct(t)[0] for t in ts) / len(ts) if ts else None)
        grid.append(row)
    return scenes, cats, grid


def category_totals(results: dict, cats: list[str]) -> tuple[list, float]:
    """Pooled effective correctness per category over all scenes, plus the grand total."""
    ts = results["trials"]
    row = []
    for c in cats:
        cs = [t for t in ts if t["category"] == c]
        row.append(sum(effective_correct(t)[0] for t in cs) / len(cs) if cs else None)
    return row, sum(effective_correct(t)[0] for t in ts) / len(ts)


def per_prompt_table(results: dict) -> list[dict]:
    rows = []
    for key, spec in results["prompts"].items():
        trials = [t for t in results["trials"] if t["key"] == key]
        if not trials:
            continue
        row = dict(key=key, category=spec["category"], prompt=spec["prompt"], n=len(trials))
        for c in CHECKS:
            row[c] = sum(t["auto_label"][c] for t in trials) / len(trials)
        row["first_pass"] = sum(first_pass_correct(spec, t) for t in trials) / len(trials)
        row["retried"] = sum(t["retried"] for t in trials)
        eff = [effective_correct(t) for t in trials]
        row["correct"] = sum(c for c, _ in eff) / len(trials)
        row["n_manual"] = sum(m for _, m in eff)
        row["plans"] = [" → ".join(t["auto_label"]["planned"] and
                                   [str(p) for p in t["auto_label"]["planned"]] or ["-"])
                        for t in trials]
        rows.append(row)
    return rows


def write_summary(results: dict, rows: list[dict], path: Path):
    m = results["meta"]
    lines = [f"# VLM decompose test" + (" — one-pass planner (original, git b0c32a51^)"
                                        if m.get("one_pass") else " — two-pass planner"), "",
             *[f"- scene `{sc['name']}`: `{sc['path']}`" for sc in m["scenes"]],
             f"- model: `{m['model']}` (temp {m['temperature']})",
             f"- trials per prompt per scene: {m['trials']}"
             + (f" — fixed subset of {m['sample']} scene x prompt pairs (seed {m['seed']})"
                if m.get("sample") else "")
             + (f" — fixed subset of {m['per_category']} scene x prompt pairs per category "
                f"(seed {m['seed']})" if m.get("per_category") else ""),
             *([f"- subset of `{m['subset_of']}`"] if m.get("subset_of") else []), "",
             "| category | key | prompt | " + " | ".join(CHECKS) + " | first pass | retried | correct | manual | plans |",
             "|" + "---|" * (len(CHECKS) + 8)]
    for r in rows:
        lines.append(f"| {r['category']} | {r['key']} | {r['prompt']} | "
                     + " | ".join(f"{r[c]:.2f}" for c in CHECKS)
                     + f" | {r['first_pass']:.2f} | {r['retried']}/{r['n']} | {r['correct']:.2f} | {r['n_manual']}/{r['n']} | {'; '.join(r['plans'])} |")
    scenes, cats, grid = scene_category_table(results)
    lines += ["", "## Per scene x category (overall correct)", "",
              "| scene | " + " | ".join(cats) + " | all |", "|" + "---|" * (len(cats) + 2)]
    for sc, row in zip(scenes, grid):
        ts = [t for t in results["trials"] if t["scene"] == sc]
        allr = sum(effective_correct(t)[0] for t in ts) / len(ts)
        lines.append(f"| {sc} | " + " | ".join("-" if v is None else f"{v:.2f}" for v in row)
                     + f" | {allr:.2f} |")
    tot, tot_all = category_totals(results, cats)
    lines.append("| **all scenes** | " + " | ".join("-" if v is None else f"**{v:.2f}**" for v in tot)
                 + f" | **{tot_all:.2f}** |")
    vague = [t for t in results["trials"] if t["auto_label"].get("resolved_by_elimination")]
    if vague:
        lines += ["", "## Vague object names (resolved by elimination, worth a manual look)", "",
                  f"{len(vague)} trial(s) named an object only vaguely; the plan was scored as if the "
                  "name meant the one object left over.", ""]
        by_scene = {}
        for t in vague:
            by_scene.setdefault(t["scene"], set()).update(
                f"'{k}' -> {v}" for k, v in t["auto_label"]["resolved_by_elimination"].items())
        lines += [f"- {sc}: {', '.join(sorted(v))}" for sc, v in by_scene.items()]
    lines += attribution_lines(results)
    lines += ["", "## Per category", ""]
    for cat in CATEGORIES:
        rs = [r for r in rows if r["category"] == cat]
        if rs:
            n = sum(r["n"] for r in rs)
            lines.append(f"- {cat}: {sum(r['correct'] * r['n'] for r in rs) / n:.2f} ({n} trials)")
    path.write_text("\n".join(lines) + "\n")


def attribution_counts(results: dict) -> tuple[collections.Counter, dict]:
    """Cause counts over auto-labelled failures, overall and per category."""
    overall, per_cat = collections.Counter(), {c: collections.Counter() for c in CATEGORIES}
    for t in results["trials"]:
        cause = t["auto_label"].get("cause")
        if cause:
            overall[cause] += 1
            per_cat[t["category"]][cause] += 1
    return overall, per_cat


def attribution_lines(results: dict) -> list[str]:
    overall, per_cat = attribution_counts(results)
    n_fail = sum(overall.values())
    real = n_fail - overall["not_a_failure"]
    if not n_fail:
        return []
    lines = ["", "## Failure attribution (auto labels)", "",
             f"{n_fail} auto-labelled failures, of which {overall['not_a_failure']} are not real failures "
             f"(the missed object was not needed). Shares below are of the {real} real failures.", "",
             "| cause | group | trials | share | meaning |", "|---|---|---|---|---|"]
    for key, (group, meaning) in CAUSES.items():
        if overall[key]:
            share = "-" if key == "not_a_failure" else f"{overall[key] / real:.0%}"
            lines.append(f"| {key} | {group} | {overall[key]} | {share} | {meaning} |")
    groups = collections.Counter()
    for key, v in overall.items():
        groups[CAUSES[key][0]] += v
    lines += ["", "By group: " + ", ".join(f"{g} {v} ({v / real:.0%})" for g, v in groups.items()
                                            if g != "none"), "",
              "| category | " + " | ".join(CAUSES) + " |", "|" + "---|" * (len(CAUSES) + 1)]
    for cat, c in per_cat.items():
        if sum(c.values()):
            lines.append(f"| {cat} | " + " | ".join(str(c[k]) for k in CAUSES) + " |")
    return lines


def plot_attribution(results: dict, path: Path):
    """Why the failed plans failed: perception vs grounding vs preference reasoning."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overall, per_cat = attribution_counts(results)
    real = sum(overall.values()) - overall["not_a_failure"]
    if not real:
        return
    TEXT, TEXT2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    # Fixed categorical order (reference palette slots 1..5); one colour per cause group.
    GROUP_COLOR = OrderedDict([("perception", "#2a78d6"), ("grounding", "#1baf7a"),
                               ("reasoning", "#eb6834"), ("mixed", "#eda100"), ("other", "#e87ba4")])
    keys = [k for k in CAUSES if k != "not_a_failure" and overall[k]]
    plt.rcParams.update({"font.size": 9, "text.color": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 0.55 * len(keys) + 2.2), facecolor=SURF,
                                 gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.55))

    # Left: every cause, share of real failures.
    y = range(len(keys))
    vals = [overall[k] for k in keys]
    ax.barh(y, vals, height=0.6, color=[GROUP_COLOR[CAUSES[k][0]] for k in keys])
    for yi, v in zip(y, vals):
        ax.text(v + real * 0.01, yi, f"{v}  ({v / real:.0%})", va="center", fontsize=9, color=TEXT)
    import textwrap
    ax.set_yticks(list(y), [textwrap.fill(CAUSES[k][1], 44) for k in keys], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.28)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"Cause of the {real} failed plans", loc="left", fontsize=10, pad=24)

    # Right: per preference category, stacked by cause group.
    cats = [c for c in CATEGORIES if sum(v for k, v in per_cat[c].items() if k != "not_a_failure")]
    left = [0] * len(cats)
    for g, color in GROUP_COLOR.items():
        widths = [sum(v for k, v in per_cat[c].items() if CAUSES[k][0] == g) for c in cats]
        if not any(widths):
            continue
        bx.barh(range(len(cats)), widths, left=left, height=0.6, color=color, label=g,
                edgecolor=SURF, linewidth=1.5)
        left = [a + b for a, b in zip(left, widths)]
    for yi, tot in enumerate(left):
        bx.text(tot + max(left) * 0.01, yi, str(tot), va="center", fontsize=9, color=TEXT)
    bx.set_yticks(range(len(cats)), cats)
    bx.invert_yaxis()
    bx.set_xlim(0, max(left) * 1.15)
    bx.grid(axis="x", color=GRID, lw=0.8)
    bx.set_axisbelow(True)
    bx.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=8,
              handlelength=1.2, borderaxespad=0.2)
    bx.set_title("Failed plans per preference category", loc="left", fontsize=10, pad=24)
    for a in (ax, bx):
        for s_ in ["top", "right", "left"]:
            a.spines[s_].set_visible(False)
        a.spines["bottom"].set_color(GRID)
        a.tick_params(length=0)
        a.set_xlabel("failed trials", color=TEXT2)

    m = results["meta"]
    groups = collections.Counter()
    for k, v in overall.items():
        groups[CAUSES[k][0]] += v
    fig.suptitle(f"Perception causes {groups['perception'] / real:.0%} of failed plans "
                 f"({groups['perception']}/{real}); {overall['perception_missing'] / real:.0%} are a missed "
                 f"object alone ({overall['perception_missing']}/{real})", x=0.01, y=1.04, ha="left", fontsize=12)
    fig.text(0.01, -0.02,
             f"{m['model']}, {len(m['scenes'])} scene(s) x {len(results['prompts'])} prompts x {m['trials']} "
             f"trial(s). A failed plan is attributed to perception when it is correct against the preference "
             f"restricted to the objects the identify pass perceived. {overall['not_a_failure']} auto-labelled "
             f"failures whose missed object was not needed are excluded.", fontsize=7.5, color=TEXT2, wrap=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def plot(results: dict, rows: list[dict], path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # fixed categorical order
    cat_color = dict(zip(CATEGORIES, SERIES))
    cmap = LinearSegmentedColormap.from_list("blue", ["#f3f7fd", "#9ec1ee", "#2a78d6", "#0f3f7a"])
    plt.rcParams.update({"font.size": 9, "text.color": TEXT, "axes.labelcolor": TEXT2,
                         "xtick.color": TEXT2, "ytick.color": TEXT})

    # Without the re-ask the first pass IS the final plan, so the column is noise.
    reask = results["meta"].get("reask_uncovered", True)
    cols = CHECKS + (["first_pass"] if reask else []) + ["correct"]
    col_names = (["objects\nidentified", "object\nset", "order", "count", "tray\ndest."]
                 + (["first pass\n(pre-retry)"] if reask else []) + ["overall\ncorrect"])
    scenes, sc_cats, sc_grid = scene_category_table(results)
    multi = len(scenes) > 1
    fig = plt.figure(figsize=(13, 0.34 * len(rows) + 2.4 + (0.3 * len(scenes) + 1.6 if multi else 0)),
                     facecolor="#fcfcfb")
    gs = fig.add_gridspec(2 if multi else 1, 2, width_ratios=[2.3, 1], wspace=0.35, hspace=0.12,
                          height_ratios=[0.34 * len(rows), 0.3 * (len(scenes) + 1) + 0.6] if multi else None)
    ax = fig.add_subplot(gs[0, 0])
    data = [[r[c] for c in cols] for r in rows]
    ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            v = r[c]
            ax.text(j, i, f"{v:.0%}" if j < len(cols) - 1 else
                    f"{v:.0%}" + ("*" if r["n_manual"] else ""),
                    ha="center", va="center", fontsize=8,
                    color="#ffffff" if v > 0.55 else TEXT)
    ax.set_xticks(range(len(cols)), col_names)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), [f"{r['prompt']}" if len(r['prompt']) < 72 else r['prompt'][:69] + "…"
                                     for r in rows], fontsize=8)
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color(TEXT)
    if reask:
        ax.axvline(len(cols) - 2.5, color="#fcfcfb", lw=3)
    ax.axvline(len(cols) - 1.5, color="#fcfcfb", lw=3)
    # Category bands: a coloured strip + separators between groups.
    prev = None
    for i, r in enumerate(rows):
        ax.add_patch(plt.Rectangle((-0.5 - 0.22, i - 0.5), 0.16, 1, color=cat_color[r["category"]],
                                   clip_on=False, lw=0))
        if prev is not None and r["category"] != prev:
            ax.axhline(i - 0.5, color="#fcfcfb", lw=3)
        prev = r["category"]
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    # Right: per-category overall pass rate.
    bx = fig.add_subplot(gs[0, 1])
    cats = [c for c in CATEGORIES if any(r["category"] == c for r in rows)]
    rates, ns = [], []
    for c in cats:
        rs = [r for r in rows if r["category"] == c]
        n = sum(r["n"] for r in rs)
        rates.append(sum(r["correct"] * r["n"] for r in rs) / n)
        ns.append(n)
    y = range(len(cats))
    bx.barh(y, rates, height=0.55, color=[cat_color[c] for c in cats])
    for yi, v, n in zip(y, rates, ns):
        bx.text(v + 0.02, yi, f"{v:.0%}  (n={n})", va="center", color=TEXT, fontsize=9)
    bx.set_yticks(list(y), cats)
    bx.invert_yaxis()
    bx.set_xlim(0, 1.25)
    bx.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    bx.grid(axis="x", color=GRID, lw=0.8)
    bx.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        bx.spines[s].set_visible(False)
    bx.spines["bottom"].set_color(GRID)
    bx.tick_params(length=0)
    bx.set_title("Overall correct by preference category", loc="left", color=TEXT, fontsize=10)

    # Bottom (multi-scene only): does the result hold across scenes?
    if multi:
        sx = fig.add_subplot(gs[1, :])
        grid = sc_grid + [category_totals(results, sc_cats)[0]]
        vals = [[float("nan") if v is None else v for v in row] for row in grid]
        sx.imshow(vals, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                sx.text(j, i, "-" if v is None else f"{v:.0%}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if (v or 0) > 0.55 else TEXT,
                        fontweight="bold" if i == len(scenes) else "normal")
        sx.axhline(len(scenes) - 0.5, color="#fcfcfb", lw=3)
        sx.set_xticks(range(len(sc_cats)), sc_cats)
        sx.xaxis.tick_top()
        sx.set_yticks(range(len(grid)), scenes + ["all scenes"], fontsize=8)
        sx.get_yticklabels()[-1].set_fontweight("bold")
        for s_ in sx.spines.values():
            s_.set_visible(False)
        sx.tick_params(length=0)
        sx.set_title("Overall correct per scene x category", loc="left", color=TEXT, fontsize=10, pad=22)

    m = results["meta"]
    n_manual = sum(r["n_manual"] for r in rows)
    n_all = sum(r["n"] for r in rows)
    mode = "One-pass (original)" if results["meta"].get("one_pass") else "Two-pass"
    fig.suptitle(f"{mode} VLM decomposition on {len(scenes)} initial scene{'s' * multi} "
                 f"(pouch, banana, toy)",
                 x=0.01, ha="left", fontsize=12, color=TEXT)
    fig.text(0.01, 0.005,
             f"{m['model']}, temp {m['temperature']}, {m['trials']} trial(s)/prompt/scene, "
             f"re-ask {'on' if m.get('reask_uncovered', True) else 'off'} · "
             f"cells = fraction of trials passing each auto check · "
             + ("first pass = plan before the uncovered-object re-ask · " if reask else "")
             + f"overall uses manual labels "
             f"where set (* = row has manual labels; {n_manual}/{n_all} trials manually labelled)",
             fontsize=7.5, color=TEXT2)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def report(results_path: Path):
    results = json.loads(results_path.read_text())
    normalise(results)
    # Re-apply the auto labeller so --replot picks up checker fixes; manual labels untouched.
    for t in results["trials"]:
        spec = results["prompts"][t["key"]]
        t["auto_label"] = auto_label(spec, t["identified"], t["subtasks"], t["error"])
        t["auto_label"]["cause"] = failure_cause(spec, t)
    results_path.write_text(json.dumps(results, indent=2))
    rows = per_prompt_table(results)
    write_summary(results, rows, results_path.parent / "summary.md")
    plot(results, rows, results_path.parent / "decompose_results.png")
    plot_attribution(results, results_path.parent / "failure_attribution.png")
    print(f"Wrote {results_path.parent / 'summary.md'}, decompose_results.png, failure_attribution.png")
    for r in rows:
        print(f"  {r['category']:<12} {r['key']:<22} correct {r['correct']:.0%}  plans: {'; '.join(r['plans'])}")


def main():
    ap = argparse.ArgumentParser(description="Two-pass VLM decomposition tester")
    ap.add_argument("--image", nargs="+", default=None,
                    help=f"Initial scene(s) with all three objects visible (default {DEFAULT_IMAGE})")
    ap.add_argument("--scene-dir", default=None, help="Use every .png/.jpg in this folder as a scene")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temp", type=float, default=0.1, help="Orchestrator default vlm_temperature")
    ap.add_argument("--trials", type=int, default=3, help="Repeats per prompt (sampling is stochastic)")
    ap.add_argument("--categories", default=None, help=f"Comma list from {CATEGORIES}")
    ap.add_argument("--redecompose", default=None,
                    help="Re-run only the decompose pass of a two-pass results.json, on its recorded "
                         "identify outputs (new output folder)")
    ap.add_argument("--role-identify", action="store_true",
                    help="Two-pass: the identify pass labels each item's role; only objects (and "
                         "surfaces as destinations) reach the planner")
    ap.add_argument("--ignored-group", action="store_true",
                    help="Two-pass runs and --redecompose: add an ignored_objects group to the decompose prompt "
                         "for listed items that are not movable objects")
    ap.add_argument("--robot-parts-rule", action="store_true",
                    help="With --redecompose: add the line telling the planner robot parts in the "
                         "object list go to excluded_objects")
    ap.add_argument("--no-fixture-filter", action="store_true",
                    help="Turn off the orchestrator's vlm_filter_fixtures (fixtures out of the object "
                         "list, fixture sub-tasks dropped) for two-pass runs")
    ap.add_argument("--one-pass", action="store_true",
                    help="The original single-call planner (prompt from git b0c32a51^): objects and "
                         "sub-tasks in one call, no identify pass")
    ap.add_argument("--reask", action="store_true",
                    help="Enable the orchestrator's uncovered-object re-ask (off by default, as in the planner)")
    ap.add_argument("--out-root", default="vlm_decompose_test")
    ap.add_argument("--replot", default=None, help="Rebuild summary + plot from an existing results.json")
    ap.add_argument("--per-category", type=int, default=None,
                    help="Only N scene x prompt pairs per preference category, split evenly over its "
                         "prompts (equal categories); with --replot, subset an existing run into <run>_cat<N>/")
    ap.add_argument("--sample", type=int, default=None,
                    help="Only ~N scene x prompt pairs (for manual review); with --replot, copy "
                         "that subset of the existing run into <run>_sample<N>/")
    ap.add_argument("--seed", type=int, default=0, help="Seed for --sample")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.redecompose:
        results_path = redecompose(args)
    elif args.replot:
        results_path = Path(args.replot)
        if args.sample or args.per_category:
            results_path = subset_results(results_path, args.sample, args.seed, args.per_category)
    else:
        results_path = run(args)
    report(results_path)


if __name__ == "__main__":
    main()
