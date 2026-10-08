"""
Offline replay of the orchestrator's retry round (round 2) on real runs.

For every task under runs/ that went to a second round, this takes the saved
round-1 final-evaluation frame, the prompt, and the sub-tasks round 1 verified
complete (from task.log), then:

    1. runs the final evaluation (survey -> judge) on that frame, as the robot did
    2. plans round 2 BOTH ways against the same frame and the same retry context:
         old       full re-decompose (fresh identify pass) with the uncovered-object
                   re-ask on -- the behaviour before targeted_retry_decompose
         targeted  plan only over the survey's objects not yet at a destination;
                   stop when there are none (targeted_retry_decompose=True)

Nothing is executed on the robot. Each case also carries an empty manual_label
block, since whether a round-2 plan is right depends on the scene in the frame.

Usage (inside the lerobot-so101 container, cwd /workspace):
    python vlm_retry_replay.py
    python vlm_retry_replay.py --runs runs/run_2026-09-1*

Outputs (vlm_retry_replay/<stamp>/): results.json, summary.md, frames/<case>.png
"""

import argparse
import glob
import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

from PIL import Image

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def find_cases(run_globs: list[str]) -> list[dict]:
    """Tasks whose task.log shows a round 2, with what round 2 needs to be replayed."""
    cases = []
    logs = sorted({p for g in run_globs for p in glob.glob(f"{g}/*/task.log")})
    for log in logs:
        lines = Path(log).read_text(errors="replace").splitlines()
        r2 = next((i for i, l in enumerate(lines) if "Decomposing high-level prompt (round 2)" in l), None)
        if r2 is None:
            continue
        task_dir = Path(log).parent
        frame = sorted(task_dir.glob("final_eval_2*.png"))
        if not frame:
            continue
        prompt = re.search(r"'(.*)'", lines[r2]).group(1)
        carried = next((l for l in reversed(lines[:r2]) if "Carrying" in l and "completed" in l), "")
        m = re.search(r"\[(.*)\]", carried)
        completed = re.findall(r"'([^']*)'", m.group(1)) if m else []
        # What round 2 actually planned on the robot, for reference.
        old_plan, started = [], False
        for l in lines[r2:]:
            if "Sub-task queue" in l:
                started = True
                continue
            s = re.search(r"strator\.py:\d+\s+\d+\. (.*)", l)
            if started and s:
                old_plan.append(s.group(1))
            elif started:
                break
        name = f"{task_dir.parent.name[len('run_'):]}_{task_dir.name}"
        cases.append(dict(case=name, task_dir=str(task_dir), frame=str(frame[0]), prompt=prompt,
                          completed=completed, logged_round2_plan=old_plan))
    return cases


def main():
    ap = argparse.ArgumentParser(description="Replay orchestrator retry rounds offline")
    ap.add_argument("--runs", nargs="+", default=["runs/run_*"], help="Run dir globs")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--out-root", default="vlm_retry_replay")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import vlm_robot_orchestrator as orch

    cases = find_cases(args.runs)
    logging.info(f"{len(cases)} task(s) with a round 2")
    out_dir = Path(args.out_root) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    planner = orch.VLMPlanner(args.model, temperature=args.temp, two_pass=True)

    results = dict(meta=dict(model=args.model, temperature=args.temp,
                             created=datetime.now().isoformat(timespec="seconds")),
                   cases=[])
    for i, c in enumerate(cases, 1):
        shutil.copy(c["frame"], out_dir / "frames" / f"{c['case']}.png")
        frame = Image.open(c["frame"]).convert("RGB")
        verdict = planner.evaluate_final(c["prompt"], frame)
        survey = verdict.get("surveyed") or []
        ctx = orch.RETRY_DECOMPOSE_CONTEXT.format(
            reason=verdict.get("reason") or "(no reason given)",
            completed=orch._format_completed_context(c["completed"]),
        )

        def plan(**kw):
            try:
                return planner.decompose(c["prompt"], frame, **kw)
            except ValueError as e:
                return f"PARSE ERROR: {e}"

        planner.reask_uncovered = True
        old = plan(extra_context=ctx)
        planner.reask_uncovered = False
        remaining, destinations = orch.remaining_from_survey(survey)
        if not survey:
            targeted, objects = "NO SURVEY (falls back to full re-decompose)", None
        elif not remaining:
            targeted, objects = "STOP (survey: nothing off a destination)", None
        else:
            objects = remaining + destinations + ["table"]
            targeted = plan(extra_context=ctx + "\n\n" + orch.TARGETED_RETRY_NOTE, objects=objects)

        row = dict(c, judge_success=verdict.get("success"), judge_reason=verdict.get("reason"),
                   survey=[{k: r.get(k) for k in ("object", "support")} for r in survey],
                   remaining=remaining, targeted_objects=objects, old_plan=old, targeted_plan=targeted,
                   manual_label=dict(old_correct=None, targeted_correct=None, notes=""))
        results["cases"].append(row)
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))
        logging.info(f"[{i}/{len(cases)}] {c['case']} '{c['prompt']}'\n"
                     f"    survey remaining={remaining}\n    old      = {old}\n    targeted = {targeted}")

    lines = ["# Retry-round replay", "", f"- model: `{args.model}` (temp {args.temp})", "",
             "| case | prompt | completed in round 1 | survey: not at destination | old round 2 | targeted round 2 |",
             "|---|---|---|---|---|---|"]
    for r in results["cases"]:
        fmt = lambda p: p if isinstance(p, str) else ("; ".join(p) or "(empty)")
        lines.append(f"| {r['case']} | {r['prompt']} | {'; '.join(r['completed']) or '-'} | "
                     f"{', '.join(r['remaining']) or '-'} | {fmt(r['old_plan'])} | {fmt(r['targeted_plan'])} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_dir}/results.json and summary.md")


if __name__ == "__main__":
    main()
