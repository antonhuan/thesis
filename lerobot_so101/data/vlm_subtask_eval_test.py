"""
Sub-task evaluation tester for the orchestrator's per-attempt success judge.

Replays every sub-task attempt the orchestrator recorded under runs/ through the
SAME evaluation call the robot uses -- what _run_high_level_task_body does after
each VLA episode:

    planner.evaluate(current_instruction, clip, fps=clip_fps, extra_context=monitor_reason)
        -> {"success": bool, "failure_type": "VOCAB" | "RETRY", "reason": str}

No evaluation logic is duplicated: the planner is called as-is and generate() is
wrapped only to record the raw output.

Attempts are rebuilt from the run folders (--build-attempts):
    * the clip the evaluator saw -- older runs saved it as eval_task<K>_<stamp>_NN.png
      frames, newer ones as one eval_task<K>_<stamp>_recap.png grid, which is cut back
      into its frames (the top 15 px of each tile carry a time label, the rest is the
      original frame);
    * from task.log: the sub-task text as executed (after any label refinement), the
      clip's frame rate, the monitor's stop reason (passed to the evaluator as
      extra_context), the no-movement flag, and the pipeline's own verdict at the time.

Ground truth per attempt (vlm_subtask_eval_attempts.json, "label" -- EDIT ME):
    target     banana / toy / pouch: the object the sub-task names
    success    true if the target ends the clip on the tray -- inside it, or resting
               partly on it (the same loose rule the final evaluation uses) -- else false
    unclear    true when the last frame cannot settle it; such attempts are not scored
    notes      free text
The evaluator's failure_type (VOCAB vs RETRY) is recorded but not scored yet.

Usage (inside the lerobot-so101 container, cwd /workspace):
    python vlm_subtask_eval_test.py --build-attempts      # (re)build the attempt list, keeps labels
    python vlm_subtask_eval_test.py --sample 200          # fixed, balanced ~200-attempt subset
    python vlm_subtask_eval_test.py --replot vlm_subtask_eval_test/<stamp>/results.json

Outputs (vlm_subtask_eval_test/<stamp>/): results.json, summary.md
"""

import argparse
import collections
import json
import logging
import math
import random
import re
import time
from datetime import datetime
from pathlib import Path

from PIL import Image

from vlm_decompose_test import OBJECTS, canonical

ATTEMPTS_FILE = "vlm_subtask_eval_attempts.json"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
RECAP_LABEL_PX = 15  # _render_clip_recap draws the time label over each tile's top rows

_EXEC = re.compile(r"\[(\d+)\] Executing sub-task: '(.*)' \(\d+ retries left\)")
_OBS = re.compile(r"Evaluation observation: video clip \((\d+) frames over ([\d.]+)s")
_RECAP = re.compile(r"Saved VLM eval recap \((\d+) frames\): (\S+_recap\.png)")
_VERDICT = re.compile(r"VLM judgement: success=(True|False) \| (?:failure_type=(\w+) \| )?(.*)")
_MONITOR = re.compile(r"VLM monitor stopped the episode: (.*)|\[VLM MONITOR\] STOP after \d+ check\(s\): (.*)")
_PROMPT = re.compile(r"Decomposing high-level prompt \(round \d+\): '(.*)'")


# --------------------------------------------------------------------------- #
# Attempts: rebuilt from the run folders
# --------------------------------------------------------------------------- #
def parse_task_log(task_dir: Path) -> list[dict]:
    """One dict per evaluated attempt, in log order."""
    attempts, cur, prompt = [], None, None
    for line in (task_dir / "task.log").read_text(errors="ignore").splitlines():
        if m := _PROMPT.search(line):
            prompt = m.group(1)
        elif m := _EXEC.search(line):
            cur = dict(task_num=int(m.group(1)), instruction=m.group(2), prompt=prompt,
                       monitor_context=None, no_movement=False, recap=None)
        elif cur is None:
            continue
        elif m := _MONITOR.search(line):
            cur["monitor_context"] = m.group(1) or m.group(2)
        elif "No movement detected" in line:
            cur["no_movement"] = True
        elif m := _RECAP.search(line):
            cur["recap"], cur["n_frames"] = m.group(2), int(m.group(1))
        elif m := _OBS.search(line):
            cur["n_frames"], cur["span_s"] = int(m.group(1)), float(m.group(2))
        elif m := _VERDICT.search(line):
            if "span_s" in cur:  # video-clip evaluations only
                cur["logged"] = dict(success=m.group(1) == "True", failure_type=m.group(2),
                                     reason=m.group(3))
                attempts.append(cur)
            cur = dict(cur, monitor_context=None, no_movement=False, recap=None)
            cur.pop("span_s", None)
    return attempts


def build_attempts(runs_root: Path = Path("runs")) -> list[dict]:
    out = []
    for task_dir in sorted(p for p in runs_root.glob("*/*/") if (p / "task.log").exists()):
        logged = parse_task_log(task_dir)
        # Older runs: clip frames on disk, matched to the log by order per sub-task number.
        frame_clips = collections.defaultdict(list)
        for f in sorted(task_dir.glob("eval_task*_[0-9][0-9].png")):
            k, stamp = re.match(r"eval_task(\d+)_(\d{8}_\d{6})_\d\d\.png", f.name).groups()
            frame_clips[int(k)].append(stamp)
        order = {k: sorted(set(v)) for k, v in frame_clips.items()}
        used = collections.Counter()
        for a in logged:
            k = a["task_num"]
            if a.get("recap"):
                src = dict(kind="recap", path=a.pop("recap"))
            elif used[k] < len(order.get(k, [])):
                stamp = order[k][used[k]]
                used[k] += 1
                src = dict(kind="frames", paths=[str(p) for p in sorted(
                    task_dir.glob(f"eval_task{k}_{stamp}_[0-9][0-9].png"))])
            else:
                continue  # frames not saved (save_frames off)
            a.pop("recap", None)
            run, sub = task_dir.parts[-2].removeprefix("run_"), task_dir.parts[-1]
            a.update(id=f"{run}_{sub}_t{k}_{len([x for x in out if x['id'].startswith(f'{run}_{sub}_t{k}_')])}",
                     task_dir=str(task_dir), clip=src,
                     fps=round((a["n_frames"] - 1) / a["span_s"], 3) if a["span_s"] > 0 else None)
            out.append(a)
    return out


def load_clip(att: dict) -> list[Image.Image]:
    """The frames the evaluator saw (recap tiles cut back out; label rows kept as-is)."""
    src = att["clip"]
    if src["kind"] == "frames":
        return [Image.open(p).convert("RGB") for p in src["paths"]]
    sheet = Image.open(src["path"]).convert("RGB")
    n = att["n_frames"]
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    tw, th = sheet.width // cols, sheet.height // rows
    frames = []
    for i in range(n):
        tile = sheet.crop(((i % cols) * tw, (i // cols) * th, (i % cols + 1) * tw, (i // cols + 1) * th))
        # Blank the recap's time label so the evaluator gets no text it never saw live.
        tile.paste((0, 0, 0), (0, 0, tile.width, RECAP_LABEL_PX + 1))
        frames.append(tile)
    return frames


def target_of(instruction: str) -> str | None:
    """The object the sub-task moves: the first object named before the destination."""
    head = re.split(r"\b(on|onto|in|into|to)\s+the\s+(tray|box)", instruction.lower())[0]
    return canonical(head) or canonical(instruction)


def cmd_build(path: Path):
    old = {a["id"]: a.get("label") for a in json.loads(path.read_text())["attempts"]} \
        if path.exists() else {}
    attempts = build_attempts()
    for a in attempts:
        a["label"] = old.get(a["id"]) or dict(target=target_of(a["instruction"]), success=None,
                                               unclear=False, notes="")
    path.write_text(json.dumps(dict(
        help="Per attempt, label.success = does the target object end the clip on the tray "
             "(inside, or resting partly on it)? label.unclear = true to leave it unscored. "
             "Rebuild with --build-attempts keeps labels by id.",
        attempts=attempts), indent=1) + "\n")
    kinds = collections.Counter(a["clip"]["kind"] for a in attempts)
    print(f"{len(attempts)} attempts ({dict(kinds)}), "
          f"{sum(a['label']['success'] is not None for a in attempts)} labelled -> {path}")


# --------------------------------------------------------------------------- #
# Sampling, running, scoring
# --------------------------------------------------------------------------- #
def scorable(att: dict) -> bool:
    L = att["label"]
    return not L.get("unclear") and L.get("success") is not None and L.get("target") in OBJECTS


def already_done(att: dict) -> bool:
    """The target was on the tray and the arm did not move: the end state is right, but
    nothing was attempted. Scored as success, and reported as its own group."""
    return bool(att["no_movement"] and att["label"]["success"])


def sample_attempts(attempts: list[dict], n: int, seed: int = 0) -> list[dict]:
    """A fixed subset of ~n scorable attempts: every failure first (they are the rarer
    class), then successes spread over sub-task texts; same labels + seed -> same subset."""
    rng = random.Random(seed)
    pool = [a for a in attempts if scorable(a)]
    fails = [a for a in pool if not a["label"]["success"]]
    wins = [a for a in pool if a["label"]["success"]]
    rng.shuffle(fails)
    rng.shuffle(wins)
    picked = fails[:n]
    use = collections.Counter(a["instruction"] for a in picked)
    while len(picked) < n and wins:
        a = min(wins, key=lambda a: use[a["instruction"]])  # ties: shuffled order
        wins.remove(a)
        use[a["instruction"]] += 1
        picked.append(a)
    return sorted(picked, key=lambda a: a["id"])


def score(att: dict, success: bool | None) -> dict:
    s = dict(expected=att["label"]["success"] if scorable(att) else None)
    s["correct"] = None if s["expected"] is None else success is not None and success == s["expected"]
    s["already_done"] = already_done(att)
    return s


def run(args) -> Path:
    import vlm_robot_orchestrator as orch

    raw: list[dict] = []
    real = orch.generate

    def recording(model, processor, messages, *a, **kw):
        t0 = time.time()
        out = real(model, processor, messages, *a, **kw)
        raw.append(dict(output=out, latency_s=round(time.time() - t0, 2)))
        return out

    orch.generate = recording
    attempts = json.loads(Path(args.attempts_file).read_text())["attempts"]
    cases = sample_attempts(attempts, args.sample, args.seed) if args.sample \
        else [a for a in attempts if scorable(a)]
    out_dir = Path(args.out_root) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    planner = orch.VLMPlanner(args.model, temperature=args.temp)
    results = dict(meta=dict(attempts_file=args.attempts_file, model=args.model, precheck=args.precheck,
                             temperature=args.temp, sample=args.sample, seed=args.seed,
                             created=datetime.now().isoformat(timespec="seconds")),
                   trials=[])
    for i, att in enumerate(cases, 1):
        raw.clear()
        error, verdict = None, {}
        clip = load_clip(att)
        t0 = time.time()
        pre = None
        if args.precheck:  # the orchestrator's pre-check (subtask_done_precheck); run on
            # every attempt so the every-attempt variant can be scored too
            try:
                pre = planner.precheck_subtask_done(att["instruction"], clip[0], clip[-1])
            except Exception as e:
                pre = dict(done=False, reason=f"error: {type(e).__name__}: {e}")
        try:  # exactly the orchestrator's call after an episode
            verdict = planner.evaluate(att["instruction"], clip, fps=att["fps"],
                                       extra_context=att["monitor_context"])
        except Exception as e:  # unparseable output is a result, not a crash
            error = f"{type(e).__name__}: {e}"
        success = bool(verdict["success"]) if "success" in verdict else None
        success_eval = success
        if pre is not None and pre["done"] and att["no_movement"]:
            # As the orchestrator: pre-check only when the arm did not move. The
            # evaluator is still run above so it can be scored alone from the same run.
            success, verdict = True, dict(reason=f"Pre-check: {pre['reason']}.")
        failure_type = verdict.get("failure_type", "RETRY") if success is False else None
        if att["no_movement"] and success is False:
            failure_type = "VOCAB"  # the orchestrator's override
        trial = dict(id=att["id"], instruction=att["instruction"], target=att["label"]["target"],
                     no_movement=att["no_movement"], monitor_context=att["monitor_context"],
                     n_frames=len(clip), fps=att["fps"], success=success,
                     failure_type=failure_type, reason=verdict.get("reason", ""), error=error,
                     success_eval=success_eval,
                     success_all=(True if pre and pre["done"] else success_eval)
                     if pre is not None else None,
                     pre_done=pre["done"] if pre is not None else None,
                     precheck=None if pre is None else {k: pre.get(k) for k in
                                                        ("done", "reason", "targets", "findings")},
                     logged=att["logged"], latency_s=round(time.time() - t0, 2), raw=list(raw))
        trial["score"] = score(att, success)
        results["trials"].append(trial)
        sc = trial["score"]
        logging.info(f"[{i}/{len(cases)}] {att['id']} '{att['instruction']}': expected={sc['expected']} "
                     f"got={success} {'OK' if sc['correct'] else 'WRONG'}")
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    orch.generate = real
    return out_dir / "results.json"


def stats(ts: list[dict], verdict_key: str = "success") -> dict:
    sc = [t for t in ts if t["score"]["expected"] is not None]
    pos = [t for t in sc if t["score"]["expected"]]
    neg = [t for t in sc if not t["score"]["expected"]]
    ok = lambda t: t[verdict_key] is not None and t[verdict_key] == t["score"]["expected"]
    return dict(n=len(sc), acc=sum(map(ok, sc)) / len(sc) if sc else None,
                fn=sum(not t[verdict_key] for t in pos), n_pos=len(pos),
                fp=sum(bool(t[verdict_key]) for t in neg), n_neg=len(neg))


def write_summary(results: dict, path: Path):
    ts, m = results["trials"], results["meta"]
    for t in ts:
        t["logged_success"] = t["logged"]["success"]
    fmt = lambda v: "-" if v is None else f"{v:.2f}"
    row = lambda label, s: (f"| {label} | {s['n']} | {fmt(s['acc'])} | {s['fn']}/{s['n_pos']} | "
                            f"{s['fp']}/{s['n_neg']} |")
    head = ["| {} | n | correct | false fail | false pass |", "|---|---|---|---|---|"]
    lines = ["# VLM sub-task evaluation test", "",
             f"- attempts: `{m['attempts_file']}`"
             + (f" — fixed subset of {m['sample']} (seed {m['seed']})" if m.get("sample") else ""),
             f"- model: `{m['model']}` (temp {m['temperature']}); planner.evaluate on the saved clip, "
             "with the clip's fps and the monitor's stop reason, as the orchestrator calls it",
             "", "false fail = object on the tray but judged failed (a needless retry); "
             "false pass = not on the tray but judged done (the task moves on without it).", "",
             "## Overall", "", head[0].format("verdict"), head[1],
             row("this run" + (" (pre-check when the arm did not move, then evaluator)"
                                if m.get("precheck") else ""), stats(ts)),
             *([row("evaluator alone (same run)", stats(ts, "success_eval")),
                row("pre-check on every attempt (rejected)", stats(ts, "success_all")),
                row("pre-check alone, as 'object on the tray?'", stats(ts, "pre_done"))]
               if m.get("precheck") else []),
             row("logged at the time (older prompts/models)", stats(ts, "logged_success")),
             "", "## By group", "", head[0].format("group"), head[1]]
    for label, sub in (("object moved this attempt", [t for t in ts if not t["score"]["already_done"]]),
                       ("already on the tray, arm did not move", [t for t in ts if t["score"]["already_done"]]),
                       *[(f"target: {o}", [t for t in ts if t["target"] == o]) for o in OBJECTS]):
        if sub:
            lines.append(row(label, stats(sub)))
    lines += ["", "## By sub-task text", "", head[0].format("instruction"), head[1]]
    for ins, n in collections.Counter(t["instruction"] for t in ts).most_common():
        lines.append(row(ins, stats([t for t in ts if t["instruction"] == ins])))
    ft = collections.Counter((t["score"]["expected"], t["failure_type"]) for t in ts if t["success"] is False)
    lines += ["", "## failure_type on judged failures (not scored)", "",
              "| truly failed? | failure_type | n |", "|---|---|---|"]
    lines += [f"| {'yes' if not e else 'no (false fail)'} | {f} | {n} |" for (e, f), n in sorted(ft.items(), key=str)]
    wrong = [t for t in ts if t["score"]["correct"] is False]
    lines += ["", "## Wrong verdicts", "", "| id | instruction | expected | got | reason |", "|---|---|---|---|---|"]
    lines += [f"| {t['id']} | {t['instruction']} | {t['score']['expected']} | {t['success']} | "
              f"{t['reason'][:160].replace('|', '/')} |" for t in wrong]
    path.write_text("\n".join(lines) + "\n")


def report(results_path: Path, attempts_file: str | None = None):
    results = json.loads(results_path.read_text())
    # Re-score against the current labels so --replot picks up label fixes.
    by_id = {a["id"]: a for a in json.loads(
        Path(attempts_file or results["meta"]["attempts_file"]).read_text())["attempts"]}
    for t in results["trials"]:
        t["score"] = score(by_id[t["id"]], t["success"])
        t["target"] = by_id[t["id"]]["label"]["target"]
    results_path.write_text(json.dumps(results, indent=2))
    write_summary(results, results_path.parent / "summary.md")
    print(f"Wrote {results_path.parent / 'summary.md'}")


def main():
    ap = argparse.ArgumentParser(description="Sub-task evaluation tester")
    ap.add_argument("--attempts-file", default=ATTEMPTS_FILE)
    ap.add_argument("--build-attempts", action="store_true",
                    help="Rebuild the attempt list from runs/ (labels are kept by id)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--temp", type=float, default=0.1, help="Orchestrator default vlm_temperature")
    ap.add_argument("--sample", type=int, default=None,
                    help="Run a fixed ~N-attempt subset (all failures, then successes spread over texts)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precheck", action="store_true",
                    help="Run the orchestrator's sub-task pre-check (object already on the tray -> "
                         "done) before the evaluator; both are recorded for every attempt")
    ap.add_argument("--out-root", default="vlm_subtask_eval_test")
    ap.add_argument("--replot", default=None, help="Rescore + rebuild the summary from a results.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.build_attempts:
        cmd_build(Path(args.attempts_file))
        return
    results_path = Path(args.replot) if args.replot else run(args)
    report(results_path, args.attempts_file if args.replot else None)


if __name__ == "__main__":
    main()
