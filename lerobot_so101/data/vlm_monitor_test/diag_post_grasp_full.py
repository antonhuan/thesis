"""Full two-trial evaluation of a POST-GRASP wrist-only check.

For every high-quality episode (same set as vlm_monitor_test.py), find the first frame
where the observed gripper closes on the first grasp (opened >= GRIPPER_OPEN_MIN above
start, then >= MARGIN below the run's open peak), and ask a forced-choice "what is
held" probe on the wrist frames c and c+1.

Held-object decision:
  two-frame  both frames name the same object -> that object; else 'unclear'
  one-frame  frame c's answer ('nothing' -> 'unclear')

Scored two ways for both trials (correct task / WRONG_OBJECT task):
  canonical  held object vs the task's PHYSICAL object (pouch / plush / banana) —
             perception only, immune to task-name paraphrases (box, roll, ...)
  compare    held object fed through the production compare pass against the task
             string — comparable to the monitor runs (includes naming mismatches)
STOP = decided object != task object ('unclear' always CONTINUE).

Run inside the lerobot-so101 container from /workspace:
  python vlm_monitor_test/diag_post_grasp_full.py
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from vlm_monitor_test import (  # noqa: E402
    WRONG_OBJECT, discover_episodes, load_ground_truth, make_task, resolve_task)
from vlm_monitor_tune import (  # noqa: E402
    GRIPPER_OPEN_MIN, MONITOR_MAX_NEW_TOKENS, _GRIPPER_IDX, _compare_content,
    default_prompts, generate, load_clip, load_model, parse_json_object)

MARGIN = 6.0
PROBE = ("This is a close-up from a camera mounted on a robot gripper; the two orange "
         "fingers are the gripper jaws. Which object is held between the jaws? Choose "
         "exactly one of: banana, brown pouch, plush moose toy, nothing. "
         'Return ONLY JSON: {"object": "<choice>", "reason": "<what you see>"}')
# Physical object behind each task word (the scene only ever holds these three).
CANON = {"pouch": "POUCH", "box": "POUCH", "purse": "POUCH", "roll": "POUCH",
         "brown": "POUCH", "brown pouch": "POUCH", "plush": "PLUSH", "moose": "PLUSH",
         "stuffed": "PLUSH", "stuffed animal": "PLUSH", "toy": "PLUSH",
         "plush toy": "PLUSH", "plush moose toy": "PLUSH", "banana": "BANANA"}


def canon(name):
    name = (name or "").strip().lower()
    if name in CANON:
        return CANON[name]
    for k, v in CANON.items():
        if k in name:
            return v
    return None


def task_object(task):
    # "put the <obj> on the tray" -> <obj>
    return task.lower().removeprefix("put the ").removesuffix(" on the tray").strip()


def first_close(src):
    start = src.initial_joints[_GRIPPER_IDX]
    peak = None
    for i, j in enumerate(src.joints):
        if j is None:
            continue
        g = j[_GRIPPER_IDX]
        peak = g if peak is None else max(peak, g)
        if peak - start >= GRIPPER_OPEN_MIN and g <= peak - MARGIN:
            return i
    return None


def compare_action(model, processor, prompts, task, held):
    ident = ({"reaching_for": "unclear", "candidates": [], "held": "yes"}
             if held == "unclear" else
             {"reaching_for": held, "candidates": [held], "held": "yes"})
    msgs = [{"role": "system", "content": [{"type": "text", "text": prompts["compare"]}]},
            {"role": "user", "content": _compare_content(task, ident)}]
    txt = generate(model, processor, msgs, max_new_tokens=MONITOR_MAX_NEW_TOKENS,
                   temperature=0.1)
    v = parse_json_object(txt) or {}
    return str(v.get("action", "CONTINUE")).upper(), v.get("reason", txt[:200])


gt = load_ground_truth("vla_failure_analysis2/labeling_worklist.csv")
episodes, _ = discover_episodes("vla_failure_test", gt, None, None)
prompts = default_prompts()
model, processor = load_model("Qwen/Qwen3-VL-8B-Instruct")

out = []
for k, (cat, name, ep_dir, obj) in enumerate(episodes, 1):
    rec = {"category": cat, "episode": name, "object_type": obj}
    src = load_clip(ep_dir)
    c = first_close(src) if src.joints is not None else None
    if c is None or c + 1 >= len(src.clip) or src.wrist is None:
        rec["skipped"] = f"no first close (c={c})"
        out.append(rec)
        print(f"[{k}/{len(episodes)}] {cat}/{name}: SKIP ({rec['skipped']})", flush=True)
        continue
    answers = []
    for i in (c, c + 1):
        msgs = [{"role": "user", "content": [{"type": "image", "image": src.wrist[i]},
                                             {"type": "text", "text": PROBE}]}]
        p = parse_json_object(generate(model, processor, msgs, max_new_tokens=128,
                                       temperature=0.1)) or {}
        answers.append({"frame": i, "object": str(p.get("object", "nothing")).lower(),
                        "reason": p.get("reason", "")})
    a0, a1 = answers[0]["object"], answers[1]["object"]
    held = {"one": "unclear" if a0 == "nothing" else a0,
            "two": a0 if (a0 == a1 and a0 != "nothing") else "unclear"}
    correct_task = resolve_task(src, cat)
    wrong_task = make_task(WRONG_OBJECT[obj])
    rec.update({"first_close": c, "answers": answers, "held": held,
                "task_correct": correct_task, "task_wrong": wrong_task})
    for trial, task in (("correct", correct_task), ("wrong", wrong_task)):
        tgt = canon(task_object(task))
        r = {}
        for rule in ("one", "two"):
            h = held[rule]
            r[f"canon_{rule}"] = "CONTINUE" if h == "unclear" or canon(h) == tgt else "STOP"
            act, why = compare_action(model, processor, prompts, task, h)
            r[f"compare_{rule}"], r[f"compare_{rule}_reason"] = act, why
        rec[trial] = r
    out.append(rec)
    print(f"[{k}/{len(episodes)}] {cat}/{name}: close@{c} {a0}/{a1} | "
          f"correct canon2={rec['correct']['canon_two']} cmp2={rec['correct']['compare_two']} | "
          f"wrong canon2={rec['wrong']['canon_two']} cmp2={rec['wrong']['compare_two']}",
          flush=True)

json.dump(out, open("vlm_monitor_test/post_grasp_full.json", "w"), indent=2)

scored = [r for r in out if "correct" in r]
print(f"\nSUMMARY — {len(scored)} scored, {len(out) - len(scored)} skipped (no first close)")
for key in ("canon_two", "canon_one", "compare_two", "compare_one"):
    fa = sum(r["correct"][key] == "STOP" for r in scored)
    mi = sum(r["wrong"][key] == "CONTINUE" for r in scored)
    n = len(scored)
    print(f"  {key:12} FA={fa:2} miss={mi:2}  specificity={100 * (1 - fa / n):5.1f}%  "
          f"recall={100 * (1 - mi / n):5.1f}%  balanced={100 * (1 - (fa + mi) / (2 * n)):5.1f}%")
print("wrote vlm_monitor_test/post_grasp_full.json")
