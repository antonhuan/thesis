"""Diagnostic: does the wrist view AFTER the gripper closes name the right object?

For the same true-FP episodes + controls as diag_true_fp.py, find the first frame where
the observed gripper has closed on the first grasp (aperture >= 6 below the run's open
peak, after opening >= 10 above start — the imminent gate's grasp test), then ask a
forced-choice "what is held" probe on the wrist frames at c, c+1, c+2.
Run inside the lerobot-so101 container from /workspace:
  python vlm_monitor_test/diag_post_grasp.py
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from vlm_monitor_tune import (  # noqa: E402
    GRIPPER_OPEN_MIN, _GRIPPER_IDX, generate, load_clip, load_model, parse_json_object)

EPS = [("true_fp", e) for e in [
    "box/017_box", "brown/083_brown", "moose/071_moose", "pouch/008_pouch",
    "pouch/078_pouch", "pouch/112_pouch", "pouch/215_pouch", "pouch/225_pouch",
    "roll/018_roll", "roll/019_roll"]] + [("control", e) for e in [
    "pouch/007_pouch", "pouch/060_pouch", "pouch/075_pouch", "pouch/107_pouch",
    "pouch/147_pouch"]]
MARGIN = 6.0
PROBE = ("This is a close-up from a camera mounted on a robot gripper; the two orange "
         "fingers are the gripper jaws. Which object is held between the jaws? Choose "
         "exactly one of: banana, brown pouch, plush moose toy, nothing. "
         'Return ONLY JSON: {"object": "<choice>", "reason": "<what you see>"}')


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


model, processor = load_model("Qwen/Qwen3-VL-8B-Instruct")
out = []
for group, ep in EPS:
    src = load_clip(os.path.join("vla_failure_test", ep))
    c = first_close(src)
    rec = {"group": group, "episode": ep, "first_close": c, "probes": []}
    for i in ([] if c is None else range(c, min(c + 3, len(src.clip)))):
        msgs = [{"role": "user", "content": [{"type": "image", "image": src.wrist[i]},
                                             {"type": "text", "text": PROBE}]}]
        txt = generate(model, processor, msgs, max_new_tokens=128, temperature=0.1)
        p = parse_json_object(txt) or {"raw": txt}
        rec["probes"].append({"frame": i, **p})
    out.append(rec)
    print(f"{group:8} {ep:18} close@{c}  "
          + "  ".join(f"f{p['frame']}={p.get('object')}" for p in rec["probes"]), flush=True)

json.dump(out, open("vlm_monitor_test/diag_post_grasp.json", "w"), indent=2)
print("wrote vlm_monitor_test/diag_post_grasp.json")
