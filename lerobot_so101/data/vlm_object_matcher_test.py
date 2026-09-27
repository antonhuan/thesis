"""
Tester for the TASK -> SCENE-OBJECT matcher, grounded on the initial scene.

The semantic monitor compares what the arm is doing against the sub-task's wording,
so a paraphrase ("put the roll on the tray" for the brown pouch) reads as a mismatch.
The matcher resolves the sub-task's object phrase to a physical object ONCE per
episode, from the INITIAL SCENE (the first front frame, before the arm moves in), so
later checks can compare against that object instead of the words.

Three modes (--mode, comma-separated; default closed,open):
  closed  the VLM chooses from a known object list (+ 'none', 'ambiguous')
  open    the VLM first lists the pickable objects it sees in the initial scene, then
          chooses among its own list (+ 'none', 'ambiguous') — no object list is given
  direct  the plainest form: the initial scene + the command, "which object is it
          referring to?", with one hint to match the command's term to the semantically
          closest object — no object list, no inventory step, no none/ambiguous options
          (a short description is requested only so free text can be scored)
Open and direct answers are free text, mapped to a physical object by keyword for
scoring only (direct: the appearance first, then the name); unmapped answers are listed.

Episodes are the same high-quality set as vlm_monitor_test.py (worklist Success=TRUE,
non_oscillation, normal), and each is matched for BOTH of that test's task strings —
the correct task (from the filenames) and the WRONG_OBJECT task — against its own
initial scene. Ground truth is PHRASE_TRUTH below (EDIT ME): each object phrase maps
to the physical object it refers to, 'NONE' (nothing pickable in the scene fits), or
'AMBIGUOUS' (more than one could fit; reported, not scored).

Usage (inside the lerobot-so101 container, from /workspace):
    python vlm_object_matcher_test.py                      # all episodes, both modes
    python vlm_object_matcher_test.py --mode open,direct --limit 20
    python vlm_object_matcher_test.py --categories box,roll,purse
"""

import argparse
import collections
import glob
import json
import os
import re
import time
from pathlib import Path

from PIL import Image

from vlm_monitor_test import (WRONG_OBJECT, discover_episodes, load_ground_truth,
                              make_task, resolve_task)
from vlm_monitor_tune import (describe_task_object, load_clip, load_model,
                              match_task_object, object_phrase, task_from_path)

# ---------------------------------------------------------------------------
# Ground truth (EDIT ME)
#
# The tabletop set holds three pickable objects: a brown zipped pouch, a plush moose
# toy and a banana (plus the pink tray, which is the destination). Keys are the object
# phrase of the task ("put the <phrase> on/in the tray/box").
# ---------------------------------------------------------------------------

POUCH, PLUSH, BANANA, NONE, AMBIGUOUS = "POUCH", "PLUSH", "BANANA", "NONE", "AMBIGUOUS"

PHRASE_TRUTH: dict[str, str] = {
    "pouch": POUCH, "brown pouch": POUCH, "box": POUCH, "purse": POUCH,
    "roll": POUCH, "pencil case": POUCH, "chocolate bar": POUCH, "chocolate": POUCH,
    "sausage": POUCH, "cylinder": POUCH, "cylindrical object": POUCH,
    "brown cylinder": POUCH, "brown cylindrical object": POUCH,
    "plush toy": PLUSH, "toy": PLUSH, "moose": PLUSH, "stuffed animal": PLUSH,
    "brown plush toy": PLUSH,
    "banana": BANANA,
    "robot arm": NONE, "table": NONE, "tray": NONE, "wires": NONE, "kumquat": NONE,
    "brown object": AMBIGUOUS, "kiwi": AMBIGUOUS,
}

# Per-scene exceptions: not every scene holds all three objects, so a phrase can name
# an ABSENT object (the WRONG_OBJECT "banana" task in some pouch scenes). Verified on the
# initial frames (no yellow pixels / visual check); 112_pouch does hold a banana,
# mostly hidden behind the moose, so it stays BANANA.
EPISODE_TRUTH: dict[tuple[str, str], str] = {
    ("147_pouch", "banana"): NONE, ("149_pouch", "banana"): NONE,
    ("199_pouch", "banana"): NONE, ("215_pouch", "banana"): NONE,
    ("219_pouch", "banana"): NONE,
}

# The closed mode's object list (what the scene holds).
CLOSED_OBJECTS = {"brown pouch": POUCH, "plush moose toy": PLUSH, "banana": BANANA}

# Keyword map from an open-vocabulary answer to the physical object (scoring only).
# Whole-word regexes, checked in order (tray, banana, plush, pouch), so "brown plush
# toy" is the plush and "pencil case" the pouch. The pouch list includes the loose names
# the VLM gives the brown pouch in open mode (sausage, chocolate bar, pen, hot dog, ...).
_OPEN_KEYWORDS = [
    # The destination tray is never the object to pick; "box" alone is NOT mapped (the
    # VLM reads "put the box ..." as the pink tray, and an echoed name proves nothing).
    ("TRAY", r"tray|pink|container|clear"),
    (BANANA, r"banana|yellow"),
    (PLUSH, r"moose|plush|stuffed|toys?|animal|deer|reindeer|elk|fur\w*|fuzzy|antlers?|kiwi"),
    (POUCH, r"pouch|pencil|case|bag|purse|wallet|leather|cylind\w*|zip\w*|sausage|"
            r"chocolate|bar|roll|hot ?dog|pens?|tube"),
]


def canon_open(name: str) -> str | None:
    n = (name or "").strip().lower()
    if n in ("none", ""):
        return NONE
    if n == "ambiguous":
        return AMBIGUOUS
    for obj, pat in _OPEN_KEYWORDS:
        if re.search(rf"\b(?:{pat})\b", n):
            return obj
    return None


def canon_direct(m: dict) -> str | None:
    """Direct-mode answer -> physical object: its APPEARANCE first (the VLM often just
    echoes the command's word as the name — 'roll' -> 'roll' — which says nothing about
    what it looked at), else the name."""
    obj = (m.get("object") or "").strip().lower()
    if obj in ("none", "ambiguous", ""):
        return canon_open(obj)
    return canon_open(own_appearance(m.get("appearance", ""))) or canon_open(obj)


_RELATIONAL = re.compile(r"\b(with|has|having|holding|containing|inside|next to|near|"
                         r"beside|behind|in front of|on|in|under|against|by)\b")


def own_appearance(text: str) -> str:
    """The object's own features: the appearance text cut at the first relational
    phrase ('brown, rectangular, with a banana next to it' -> 'brown, rectangular,'),
    so a neighbour mentioned in it is not scored as the object."""
    return _RELATIONAL.split((text or "").lower(), maxsplit=1)[0]


def used_name_fallback(m: dict) -> bool:
    obj = (m.get("object") or "").strip().lower()
    return obj not in ("none", "ambiguous", "") and canon_open(own_appearance(m.get("appearance", ""))) is None


def predict(mode: str, m: dict) -> str | None:
    if mode == "closed":
        return CLOSED_OBJECTS.get(m["object"]) or canon_open(m["object"])
    return canon_direct(m) if mode == "direct" else canon_open(m["object"])


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def discover(root: str, worklist: str, categories: list[str] | None) -> list[dict]:
    """One case per (episode, trial) over vlm_monitor_test.py's high-quality set: the
    correct task and the WRONG_OBJECT task, both against the episode's initial scene."""
    episodes, _ = discover_episodes(root, load_ground_truth(worklist), categories, None)
    cases = []
    for cat, name, ep_dir, obj in episodes:
        wrong = WRONG_OBJECT.get(obj, "").strip()
        src = load_clip(ep_dir)
        for trial, task in (("correct", resolve_task(src, cat)),
                            ("wrong", make_task(wrong) if wrong else "")):
            cases.append({"category": cat, "episode": name, "trial": trial,
                          "dir": ep_dir, "initial": src.paths[0], "task": task,
                          "phrase": object_phrase(task) if task else ""})
    return cases


def discover_all(root: str, categories: list[str] | None) -> list[dict]:
    """One case per episode over EVERY episode under `root` with a task in its
    filenames (its own task + initial scene; no wrong-object trial)."""
    cases = []
    for ep_dir in sorted(glob.glob(os.path.join(root, "*", "*"))):
        cat = Path(ep_dir).parent.name
        if not os.path.isdir(ep_dir) or (categories and cat not in categories):
            continue
        fronts = sorted(p for p in glob.glob(os.path.join(ep_dir, "*_front_*"))
                        if p.lower().endswith((".png", ".jpg", ".jpeg")))
        task = task_from_path(fronts[0]) if fronts else ""
        cases.append({"category": cat, "episode": Path(ep_dir).name, "trial": "correct",
                      "dir": ep_dir, "initial": fronts[0] if fronts else "", "task": task,
                      "phrase": object_phrase(task) if task else ""})
    return cases


# Phrases that literally name one of the three objects — trivial for grounding.
_LITERAL = re.compile(r"\b(pouch|banana|toy)\b")


def summarize(results: list[dict], mode: str) -> None:
    key = f"{mode}_pred"
    rs = [r for r in results if key in r]
    scored = [r for r in rs if r["truth"] not in (AMBIGUOUS, None)]
    ok = [r for r in scored if r[key] == r["truth"]]
    print(f"\n{'#' * 64}\n{mode.upper()} MODE — accuracy {len(ok)}/{len(scored)} "
          f"({100 * len(ok) / max(1, len(scored)):.1f}%) on scored phrases\n{'#' * 64}")
    for trial in ("correct", "wrong"):
        ts = [r for r in scored if r.get("trial") == trial]
        print(f"  {trial:7} task: {sum(r[key] == r['truth'] for r in ts)}/{len(ts)}")
    by_phrase = collections.defaultdict(list)
    for r in rs:
        by_phrase[(r["truth"] or "?", r["phrase"])].append(r[key])
    print(f"  {'truth':10}{'phrase':28}{'n':>4}  predictions")
    for (truth, phrase), preds in sorted(by_phrase.items()):
        c = collections.Counter(p or "?" for p in preds)
        mark = "" if truth == AMBIGUOUS else ("  ✓" if c.get(truth, 0) == len(preds) else "  ✗")
        print(f"  {truth:10}{phrase:28}{len(preds):>4}  {dict(c)}{mark}")
    conf = collections.Counter((r["truth"], r[key] or "?") for r in scored)
    labels = [POUCH, PLUSH, BANANA, NONE]
    cols = labels + ["TRAY", AMBIGUOUS, "?"]
    print(f"\n  confusion (rows = truth, cols = predicted)\n  {'':10}" +
          "".join(f"{c:>10}" for c in cols))
    for t in labels:
        print(f"  {t:10}" + "".join(f"{conf.get((t, c), 0):>10}" for c in cols))
    if mode == "direct":
        fb = [r for r in scored if used_name_fallback(r["direct_raw"])]
        print(f"\n  scored from the NAME (appearance gave no match): {len(fb)}/{len(scored)} "
              f"— {sum(r[key] == r['truth'] for r in fb)} of those correct")
    unmapped = collections.Counter(r[f"{mode}_raw"]["object"] for r in rs if r[key] is None)
    if unmapped:
        print(f"\n  unmapped answers (scored '?'): {dict(unmapped)}")


def main():
    ap = argparse.ArgumentParser(description="Task -> scene-object matcher tester")
    ap.add_argument("--root", default="vla_failure_test")
    ap.add_argument("--worklist", default="vla_failure_analysis2/labeling_worklist.csv")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--mode", default="closed,open",
                    help="Comma-separated modes: closed, open, direct (or 'all')")
    ap.add_argument("--categories", default=None, help="Comma-separated categories")
    ap.add_argument("--set", choices=["hq", "all"], default="hq",
                    help="hq: the monitor test's high-quality episodes, correct + wrong task "
                         "(default); all: every episode under --root, its own task only")
    ap.add_argument("--paraphrase-only", action="store_true",
                    help="Drop phrases that literally contain pouch / banana / toy")
    ap.add_argument("--limit", type=int, default=None, help="Cap episodes processed")
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--out", default="vlm_monitor_test/object_matcher.json")
    ap.add_argument("--rescore", default=None,
                    help="Re-score a saved results JSON with the current ground truth "
                         "(no model load) and exit")
    args = ap.parse_args()

    if args.rescore:
        results = json.load(open(args.rescore))["results"]
        for r in results:
            r["truth"] = EPISODE_TRUTH.get((r["episode"], r["phrase"]),
                                           PHRASE_TRUTH.get(r["phrase"]))
        for mode in ("closed", "open", "direct"):
            if any(f"{mode}_raw" in r for r in results):
                for r in results:
                    if f"{mode}_raw" in r:
                        r[f"{mode}_pred"] = predict(mode, r[f"{mode}_raw"])
                summarize(results, mode)
        return

    cats = [c.strip() for c in args.categories.split(",")] if args.categories else None
    eps = (discover_all(args.root, cats) if args.set == "all"
           else discover(args.root, args.worklist, cats))
    if args.paraphrase_only:
        eps = [e for e in eps if e["task"] and not _LITERAL.search(e["phrase"])]
    no_task = [e for e in eps if not e["task"]]
    eps = [e for e in eps if e["task"]][:args.limit]
    unknown = sorted({e["phrase"] for e in eps if e["phrase"] not in PHRASE_TRUTH})
    print(f"{len(eps)} case(s) = {len({e['dir'] for e in eps})} episode(s) x "
          f"(correct, wrong) task ({len(no_task)} without a task skipped).")
    if unknown:
        print(f"[WARN] phrases with no PHRASE_TRUTH entry (unscored): {unknown}")

    modes = (["closed", "open", "direct"] if args.mode == "all"
             else [m.strip() for m in args.mode.split(",") if m.strip()])
    bad = [m for m in modes if m not in ("closed", "open", "direct")]
    if bad:
        ap.error(f"unknown mode(s): {bad}")
    model, processor = load_model(args.model)
    results = []
    t0 = time.time()
    for k, e in enumerate(eps, 1):
        rec = dict(e, truth=EPISODE_TRUTH.get((e["episode"], e["phrase"]),
                                              PHRASE_TRUTH.get(e["phrase"])))
        initial = Image.open(e["initial"]).convert("RGB")
        line = (f"[{k}/{len(eps)}] {e['category']}/{e['episode']} {e['trial']:7} "
                f"'{e['phrase']}' (truth {rec['truth']})")
        for mode in modes:
            if mode == "direct":
                m = describe_task_object(model, processor, initial, e["task"], args.temp)
            else:
                m = match_task_object(model, processor, initial, e["task"],
                                      objects=list(CLOSED_OBJECTS) if mode == "closed" else None,
                                      temperature=args.temp)
            pred = predict(mode, m)
            rec[f"{mode}_raw"], rec[f"{mode}_pred"] = m, pred
            flag = "" if rec["truth"] in (AMBIGUOUS, None) or pred == rec["truth"] else " ✗"
            shown = m["object"] or "?"
            if mode == "direct":
                shown += f" [{m.get('appearance', '')}]"
            line += f" | {mode}: {shown} -> {pred}{flag}"
        results.append(rec)
        print(line, flush=True)

    for mode in modes:
        summarize(results, mode)
    print(f"\nElapsed: {time.time() - t0:.1f}s")
    if args.out:
        Path(args.out).write_text(json.dumps({"config": vars(args),
                                              "phrase_truth": PHRASE_TRUTH,
                                              "results": results}, indent=2))
        print(f"Wrote {len(results)} records to {args.out}")


if __name__ == "__main__":
    main()
