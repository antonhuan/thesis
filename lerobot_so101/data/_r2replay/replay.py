"""Replay the 19 round-2 decomposes from preference_eval_system_trials on their saved
frames: T = targeted retry (as logged), F = full perception pass with the evaluator's
reason, FN = full perception pass without it. Writes _r2replay/results.json."""
import ast
import contextlib
import io
import json
import re
from pathlib import Path

from PIL import Image

import vlm_robot_orchestrator as O


def trial_path(n):
    """The replan-on original: archived when a replan-off re-run replaced it."""
    a = Path(f"preference_eval_system_trials/archive/trial_{n:03d}.json")
    return a if a.exists() else Path(f"preference_eval_system_trials/trial_{n:03d}.json")


CASES = [12, 14, 16, 17, 20, 24, 28, 29, 34, 39, 40, 43, 44, 45, 46, 47, 50, 53, 55]
NO_REASON = O.RETRY_DECOMPOSE_CONTEXT.replace(
    " by the evaluator, who reported:\n{reason}\n{completed}", ".\n{completed}")
assert NO_REASON != O.RETRY_DECOMPOSE_CONTEXT

planner = O.VLMPlanner("Qwen/Qwen3-VL-8B-Instruct", temperature=0.1, eval_fps=2.0,
                       two_pass=True, reask_uncovered=False, filter_fixtures=True)
out = []
for n in CASES:
    t = json.load(open(trial_path(n)))
    s = t["system"]
    d = Path(s["task_dir"])
    frame = Image.open(sorted(d.glob("decompose_round2_*.png"))[0]).convert("RGB")
    log = (d / "task.log").read_text().split("ROUND 2 for")[0]
    m = re.search(r"planning over (\[.*?\])\.", log)
    retry_objects = ast.literal_eval(m.group(1)) if m else None
    completed = list(s["rounds"][0]["completed"])
    reason = s["rounds"][0]["final_reason"] or "(no reason given)"
    ctx = O.RETRY_DECOMPOSE_CONTEXT.format(
        reason=reason, completed=O._format_completed_context(completed))
    variants = {
        "T": (ctx + "\n\n" + O.TARGETED_RETRY_NOTE, retry_objects),
        "F": (ctx, None),
        "FN": (NO_REASON.format(completed=O._format_completed_context(completed)), None),
    }
    rec = {"trial": n, "prompt": t["prompt"], "logged": s["rounds"][1]["planned"]}
    for name, (context, objects) in variants.items():
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                plan = planner.decompose(t["prompt"], frame, extra_context=context,
                                         objects=objects)
        except ValueError as e:
            plan = f"PARSE_FAIL: {e}"
        rec[name] = plan
        rec[name + "_trace"] = "\n".join(l for l in buf.getvalue().splitlines()
                                         if l.startswith(("Visible", "Excluded", "Completed",
                                                          "Allowed")))
    print(f"#{n} {t['prompt']!r}\n  logged={rec['logged']}\n  T={rec['T']}\n  F={rec['F']}\n"
          f"  FN={rec['FN']}", flush=True)
    out.append(rec)
    json.dump(out, open("_r2replay/results.json", "w"), indent=1)
print("DONE")
