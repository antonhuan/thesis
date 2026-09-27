"""Diagnostic: why does the identify pass misread the pouch at the grasp moment?

For each TRUE false-positive window in imminent6.json (the correct-trial STOP window,
naming-only cases excluded) plus a few clean pouch controls, re-run the identify pass
with the views ablated and capture its raw output:
  full   initial scene + front track + wrist track (production)
  wrist  wrist track only
  front  initial scene + front track only
  probe  latest wrist frame + "what is between the jaws?" (forced choice)
Run inside the lerobot-so101 container from /workspace:
  python vlm_monitor_test/diag_true_fp.py
"""
import json
import os
import sys

sys.path.insert(0, os.getcwd())
from vlm_monitor_tune import (  # noqa: E402
    MONITOR_MAX_NEW_TOKENS, _identify_content, default_prompts, generate, load_clip,
    load_model, parse_json_object)

TRUE_FP = ["box/017_box", "brown/083_brown", "moose/071_moose", "pouch/008_pouch",
           "pouch/078_pouch", "pouch/112_pouch", "pouch/215_pouch", "pouch/225_pouch",
           "roll/018_roll", "roll/019_roll"]
CONTROLS = ["pouch/007_pouch", "pouch/060_pouch", "pouch/075_pouch", "pouch/107_pouch",
            "pouch/147_pouch"]
PROBE = ("This is a close-up from a camera mounted on a robot gripper; the two orange "
         "fingers are the gripper jaws. Which object is directly between the jaws? "
         "Choose exactly one of: banana, brown pouch, plush moose toy, nothing. "
         'Return ONLY JSON: {"object": "<choice>", "reason": "<what you see>"}')

res = json.load(open("vlm_monitor_test/imminent6.json"))
by_ep = {f"{r['category']}/{r['episode']}": r for r in res["results"] if "correct" in r}
win = res["config"]["window"]


def target_window(ep):
    tl = by_ep[ep]["correct"]["timeline"]
    stop = next((w for w in tl if w["action"] == "STOP"), None)
    return (stop or next(w for w in tl if w["action"] != "GATED"))["span"]


def ablate(content, n, keep):
    """content = [init_txt, init_img, front_hdr, (lbl, img)*n, wrist_hdr, (lbl, img)*n, q]."""
    init, front = content[0:2], content[2:3 + 2 * n]
    wrist, q = content[3 + 2 * n:4 + 4 * n], content[4 + 4 * n:]
    return {"full": content, "wrist": wrist + q, "front": init + front + q}[keep]


prompts = default_prompts()
model, processor = load_model("Qwen/Qwen3-VL-8B-Instruct")
out = []
for group, eps in (("true_fp", TRUE_FP), ("control", CONTROLS)):
    for ep in eps:
        src = load_clip(os.path.join("vla_failure_test", ep))
        span = target_window(ep)
        lo = [i for i, p in enumerate(src.paths)
              if os.path.basename(p).startswith(span.split("..")[0])][0]
        w = src.window(lo, win)
        content = _identify_content(w.clip, w.initial, 2.0, wrist=w.wrist)
        rec = {"group": group, "episode": ep, "span": span}
        for keep in ("full", "wrist", "front"):
            msgs = [{"role": "system", "content": [{"type": "text", "text": prompts["identify"]}]},
                    {"role": "user", "content": ablate(content, len(w.clip), keep)}]
            txt = generate(model, processor, msgs, max_new_tokens=MONITOR_MAX_NEW_TOKENS,
                           temperature=0.1)
            rec[keep] = parse_json_object(txt) or {"raw": txt}
        msgs = [{"role": "user", "content": [{"type": "image", "image": w.wrist[-1]},
                                             {"type": "text", "text": PROBE}]}]
        txt = generate(model, processor, msgs, max_new_tokens=128, temperature=0.1)
        rec["probe"] = parse_json_object(txt) or {"raw": txt}
        out.append(rec)
        print(f"{group:8} {ep:18} {span:14} full={rec['full'].get('reaching_for')!s:12} "
              f"wrist={rec['wrist'].get('reaching_for')!s:12} "
              f"front={rec['front'].get('reaching_for')!s:12} "
              f"probe={rec['probe'].get('object')}", flush=True)

json.dump(out, open("vlm_monitor_test/diag_true_fp.json", "w"), indent=2)
print("wrote vlm_monitor_test/diag_true_fp.json")
