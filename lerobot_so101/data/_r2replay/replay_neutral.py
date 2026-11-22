"""Neutral retry framing: no 'judged INCOMPLETE', no evaluator reason, no invariant.
NF = full perception pass, NT = targeted object list. Appends to results.json."""
import ast, contextlib, io, json, re
from pathlib import Path
from PIL import Image
import vlm_robot_orchestrator as O


def trial_path(n):
    """The replan-on original: archived when a replan-off re-run replaced it."""
    a = Path(f"preference_eval_system_trials/archive/trial_{n:03d}.json")
    return a if a.exists() else Path(f"preference_eval_system_trials/trial_{n:03d}.json")


NEUTRAL = """Part of this instruction may already have been carried out.
{done}
The camera image shows the CURRENT scene. Plan the ORIGINAL instruction against this scene from scratch:
- An object the instruction wants on the tray that is already on the tray needs no sub-task — list it in completed_objects.
- An object the instruction does not select (excluded, does not match its description, or beyond the number it asks for, counting objects already on the tray) gets no sub-task.
- If nothing remains to do, return empty allowed_objects and an empty subtasks list. That is a valid and common answer."""

def done_block(completed):
    if not completed:
        return "No sub-task is known to have been completed."
    return "These sub-tasks were already executed and verified:\n" + "\n".join(f"- {c}" for c in completed)

planner = O.VLMPlanner("Qwen/Qwen3-VL-8B-Instruct", temperature=0.1, eval_fps=2.0,
                       two_pass=True, reask_uncovered=False, filter_fixtures=True)
res = json.load(open("_r2replay/results.json"))
for rec in res:
    n = rec["trial"]
    t = json.load(open(trial_path(n))); s = t["system"]
    d = Path(s["task_dir"])
    frame = Image.open(sorted(d.glob("decompose_round2_*.png"))[0]).convert("RGB")
    log = (d / "task.log").read_text().split("ROUND 2 for")[0]
    m = re.search(r"planning over (\[.*?\])\.", log)
    retry_objects = ast.literal_eval(m.group(1)) if m else None
    ctx = NEUTRAL.format(done=done_block(s["rounds"][0]["completed"]))
    for name, objs in (("NF", None), ("NT", retry_objects)):
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                plan = planner.decompose(t["prompt"], frame, extra_context=ctx, objects=objs)
        except ValueError as e:
            plan = f"PARSE_FAIL: {e}"
        rec[name] = plan
        rec[name + "_trace"] = "\n".join(l for l in buf.getvalue().splitlines()
                                         if l.startswith(("Visible", "Excluded", "Completed", "Allowed")))
    print(f"#{n} NF={rec['NF']} NT={rec['NT']}", flush=True)
    json.dump(res, open("_r2replay/results.json", "w"), indent=1)
print("DONE")
