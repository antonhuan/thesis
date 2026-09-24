"""
Batch ACCURACY harness for the two-pass semantic monitor, scored against the
hand-labelled ground truth in vla_failure_analysis2/labeling_worklist.csv.

The test runs in two folds:

  1. HIGH-QUALITY SUBSET. Only episodes the worklist marks as clean runs are used:
        Success == TRUE                 (the pick actually succeeded)
        manual_label == non_oscillation (human said it did not oscillate)
        oscillation_class == normal     (kinematic class is normal)
     This keeps the semantic test on runs where the arm genuinely reaches for and
     grasps the right object, so a STOP is unambiguously a monitor error.

  2. TWO TRIALS PER EPISODE (correct vs. incorrect object). Each usable episode is
     scanned TWICE:
        correct  — sub-task names the object the arm is actually going for. A
                   correctly-behaving monitor should NOT stop (CONTINUE throughout).
                   A STOP here is a FALSE ALARM.
        wrong    — sub-task names a deliberately different object (from WRONG_OBJECT
                   below). The arm is reaching for something else, so the monitor
                   SHOULD stop. A STOP here is a CORRECT DETECTION; no STOP is a MISS.
                   The monitor watches the trajectory converge, so this STOP can (and
                   should) fire mid-approach, BEFORE the grasp closes.
     Together these give the monitor's specificity (1 - false-alarm rate on matching
     objects) and its recall (detection rate on mismatched objects) — a baseline
     accuracy for the semantic check.

Each episode dir (vla_failure_test/<category>/<NNN_category>/) holds an ordered run
of front/wrist frames plus actions.csv (joint states). Episodes named in the worklist
but absent on disk, or missing actions.csv (the joint gate has nothing to run on), are
skipped and counted separately.

All monitor logic (prompts, the sliding trajectory window, motion gate, thresholds,
both camera views with the wrist frame rotated to match the front, the kinematic
grasp-commit, and the convergence-based STOP) is imported from vlm_monitor_tune, so this
test always exercises the current tuned behavior — no monitor logic is duplicated here.

Modes:
  --gate-only   No model. For every sliding window, just evaluate the joint/motion
                gate and report which windows are gated and where the VLM would first
                open. The gate is object-independent, so this runs once per episode
                (no two-trial scoring). Fast + deterministic — for calibrating gate
                thresholds across the high-quality subset.
  (default)     Load the model and run both trials per episode, scoring accuracy.

Usage:
    # Fast gate sweep over the high-quality subset (no model):
    python vlm_monitor_test.py --gate-only

    # Full accuracy run over two categories, 5 episodes each:
    python vlm_monitor_test.py --categories plush,banana --limit-per-category 5

    # Everything, full accuracy run, write machine-readable results:
    python vlm_monitor_test.py --out results.json
"""

import argparse
import contextlib
import csv
import io
import json
import os
import time
from pathlib import Path

from vlm_monitor_tune import (
    DEFAULT_GRASP_CLOSE_MARGIN,
    DEFAULT_JOINT_START_THRESH,
    DEFAULT_JOINT_STILL_THRESH,
    DEFAULT_MOTION_THRESH,
    DEFAULT_WRIST_ROTATE_CCW,
    ClipSource,
    _frame_index,
    default_prompts,
    evaluate_gate,
    load_clip,
    run_scan,
    task_from_path,
)


# ---------------------------------------------------------------------------
# Wrong-object map (EDIT ME)
#
# For the incorrect-object trial, each true object_type is paired with a clearly
# DIFFERENT object the arm is NOT reaching for, so a working monitor should STOP.
# Keep each mismatch obviously distinct from its key (never a synonym): the arm
# grasps the real object, and we want the compare pass to rule it a mismatch.
#
# Keys are the `object_type` values in labeling_worklist.csv. Pre-filled for the
# object types that survive the high-quality filter; add rows here as needed. An
# episode whose object_type is missing (or blank) from this map is SKIPPED and
# listed at the end so you know to fill it in.
# ---------------------------------------------------------------------------

WRONG_OBJECT: dict[str, str] = {
    "banana":  "toy",
    "box":     "stuffed animal",
    "brown":   "banana",
    "moose":   "pouch",
    "plush":   "pouch",
    "pouch":   "banana",
    "purse":   "banana",
    "roll":    "plush toy",
    "stuffed": "pouch",
    "toy":     "pouch",
    # --- add more object types below as you extend the test set ---
    # "chocolate": "banana",
    # "kiwi":      "pencil",
    # "sausage":   "banana",
}


def make_task(obj: str) -> str:
    """The sub-task phrasing fed to the monitor for a given object."""
    return f"put the {obj} on the tray"


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def load_ground_truth(csv_path: str) -> dict[tuple[str, str], str]:
    """Read the labelling worklist and return the HIGH-QUALITY subset.

    Keeps only rows with Success==TRUE, manual_label==non_oscillation and
    oscillation_class==normal. Returns {(category, episode_name): object_type},
    where episode_id 'banana__001_banana' maps to category 'banana' and episode
    '001_banana' — i.e. the on-disk path vla_failure_test/banana/001_banana.
    """
    usable: dict[tuple[str, str], str] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("Success", "").strip().upper() != "TRUE"
                    or row.get("manual_label", "").strip().lower() != "non_oscillation"
                    or row.get("oscillation_class", "").strip().lower() != "normal"):
                continue
            ep_id = row.get("episode_id", "").strip()
            if "__" not in ep_id:
                continue
            cat, ep_name = ep_id.split("__", 1)
            usable[(cat, ep_name)] = row.get("object_type", "").strip()
    return usable


def discover_episodes(root: str, ground_truth: dict[tuple[str, str], str],
                      categories: list[str] | None,
                      limit_per_category: int | None
                      ) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str]]]:
    """Resolve usable episodes that exist on disk.

    Returns (episodes, missing), where episodes is
    [(category, episode_name, episode_dir, object_type), ...] for worklist entries
    present under `root`, sorted and filtered by `categories` / capped per category,
    and missing is [(category, episode_name), ...] labelled usable but absent on disk.
    """
    episodes: list[tuple[str, str, str, str]] = []
    missing: list[tuple[str, str]] = []
    per_cat: dict[str, int] = {}
    for (cat, ep_name), obj in sorted(ground_truth.items()):
        if categories and cat not in categories:
            continue
        ep_dir = os.path.join(root, cat, ep_name)
        if not os.path.isdir(ep_dir):
            missing.append((cat, ep_name))
            continue
        if limit_per_category is not None and per_cat.get(cat, 0) >= limit_per_category:
            continue
        per_cat[cat] = per_cat.get(cat, 0) + 1
        episodes.append((cat, ep_name, ep_dir, obj))
    return episodes, missing


def resolve_task(source: ClipSource, category: str) -> str:
    """Correct sub-task for an episode: prefer the filename's, else the category."""
    for p in (source.paths[-1], source.paths[0]):
        t = task_from_path(p)
        if t:
            return t
    return make_task(category)


def _span(paths: list[str], i: int, win: int) -> str:
    lo, hi = _frame_index(paths[i]), _frame_index(paths[i + win - 1])
    return f"{lo}..{hi}"


# ---------------------------------------------------------------------------
# Gate-only sweep (no model)
# ---------------------------------------------------------------------------

def gate_only_episode(source: ClipSource, win: int, start_thresh: float,
                      still_thresh: float, motion_thresh: float) -> dict:
    """Slide the window and record the gate decision for each, without the VLM.

    Returns per-episode stats: number of windows, how many gated, the frame span of
    the first window the VLM would open on (first non-gated), and the gate signal.
    """
    n = len(source.clip)
    windows = []
    first_open = None
    for i in range(0, n - win + 1):
        w = source.window(i, win)
        g = evaluate_gate(w, start_thresh, still_thresh, motion_thresh)
        span = _span(source.paths, i, win)
        windows.append({"span": span, "gated": g["gated"], "signal": g["signal"],
                        "dist": g["dist"], "peak": g["peak"], "motion": g["motion"]})
        if not g["gated"] and first_open is None:
            first_open = span
    n_gated = sum(1 for w in windows if w["gated"])
    return {
        "n_windows": len(windows),
        "n_gated": n_gated,
        "gate_signal": windows[0]["signal"] if windows else None,
        "first_vlm_window": first_open,
        "all_gated": n_gated == len(windows) and len(windows) > 0,
        "windows": windows,
    }


# ---------------------------------------------------------------------------
# Full monitor run (with model)
# ---------------------------------------------------------------------------

def full_episode(model, processor, prompts, source: ClipSource, task: str,
                 fps: float, temp: float, win: int, start_thresh: float,
                 still_thresh: float, motion_thresh: float,
                 stop_on_stop: bool, grasp_memory: bool = True,
                 grasp_close_margin: float = DEFAULT_GRASP_CLOSE_MARGIN) -> dict:
    """Run the full sliding-window monitor on one episode for ONE sub-task.

    Wraps run_scan (whose verbose output is suppressed) and reduces its timeline to a
    compact record: the first STOP window (if any) and its reason, the first window
    the VLM opened on, and per-window actions. The scan retires the semantic check
    once the right object is confirmed grasped (grasp commit), so post-commit windows
    show up as CONTINUE in the 'held' phase.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        timeline = run_scan(model, processor, prompts, source, task, temp,
                            fps=fps, win=win, step=1, stop_on_stop=stop_on_stop,
                            motion_thresh=motion_thresh, start_thresh=start_thresh,
                            still_thresh=still_thresh, grasp_memory=grasp_memory,
                            grasp_close_margin=grasp_close_margin)
    # timeline: list of (span, action, reason, phase)
    first_open = next((sp for sp, a, _, _ in timeline if a != "GATED"), None)
    stop = next(((sp, r) for sp, a, r, _ in timeline if a == "STOP"), None)
    committed = any(ph == "held" for _, _, _, ph in timeline)
    return {
        "task": task,
        "n_windows": len(timeline),
        "n_gated": sum(1 for _, a, _, _ in timeline if a == "GATED"),
        "n_parse_fail": sum(1 for _, a, _, _ in timeline if a == "PARSE_FAIL"),
        "first_vlm_window": first_open,
        "committed": committed,
        "stopped": stop is not None,
        "stop_window": stop[0] if stop else None,
        "stop_reason": stop[1] if stop else None,
        "timeline": [{"span": sp, "action": a, "reason": r, "phase": ph}
                     for sp, a, r, ph in timeline],
    }


def score_episode(model, processor, prompts, source: ClipSource,
                  correct_task: str, wrong_task: str, args) -> dict:
    """Run both trials (correct + wrong object) and score the monitor's calls.

    correct trial : monitor should CONTINUE — a STOP is a false alarm.
    wrong trial   : monitor should STOP     — a STOP is a correct detection.
    """
    stop_on_stop = not args.no_stop_on_first
    grasp_memory = not args.no_grasp_memory
    correct = full_episode(model, processor, prompts, source, correct_task, args.fps,
                           args.temp, args.window, args.joint_start_thresh,
                           args.joint_still_thresh, args.motion_thresh, stop_on_stop,
                           grasp_memory=grasp_memory,
                           grasp_close_margin=args.grasp_close_margin)
    wrong = full_episode(model, processor, prompts, source, wrong_task, args.fps,
                         args.temp, args.window, args.joint_start_thresh,
                         args.joint_still_thresh, args.motion_thresh, stop_on_stop,
                         grasp_memory=grasp_memory,
                         grasp_close_margin=args.grasp_close_margin)
    return {
        "correct": correct,
        "wrong": wrong,
        # A correctly-matching object should never STOP.
        "correct_pass": not correct["stopped"],
        # A mismatched object should STOP.
        "wrong_pass": wrong["stopped"],
        # The gate is object-independent; if it never opened, the VLM never got to
        # judge — the wrong trial cannot detect anything and the correct trial trivially
        # "passes". Surfaced so these episodes can be read apart from real decisions.
        "never_opened": correct["first_vlm_window"] is None,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Accuracy test for the semantic monitor vs. labelled ground truth")
    ap.add_argument("--root", default="vla_failure_test", help="Root of the episode set")
    ap.add_argument("--worklist", default="vla_failure_analysis2/labeling_worklist.csv",
                    help="Ground-truth CSV (Success/manual_label/oscillation_class labels)")
    ap.add_argument("--gate-only", action="store_true",
                    help="Skip the VLM; only sweep the motion gate (no model load, no "
                         "two-trial scoring)")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--categories", default=None,
                    help="Comma-separated category names to include (default: all usable)")
    ap.add_argument("--limit-per-category", type=int, default=None,
                    help="Cap episodes taken from each category")
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="Global cap on episodes processed")
    ap.add_argument("--view", default="front", help="Front (primary) camera marker")
    ap.add_argument("--wrist-view", default="wrist", help="Wrist (in-hand) camera marker")
    ap.add_argument("--wrist-rotate", type=float, default=DEFAULT_WRIST_ROTATE_CCW,
                    help="Degrees CCW to rotate each wrist frame at load so it lines up "
                         "with the front view (the wrist camera is mounted 90° CW). "
                         "0 disables. Must match what the tuned monitor expects. "
                         f"Default: {DEFAULT_WRIST_ROTATE_CCW}")
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--joint-start-thresh", type=float, default=DEFAULT_JOINT_START_THRESH)
    ap.add_argument("--joint-still-thresh", type=float, default=DEFAULT_JOINT_STILL_THRESH)
    ap.add_argument("--motion-thresh", type=float, default=DEFAULT_MOTION_THRESH)
    ap.add_argument("--no-stop-on-first", action="store_true",
                    help="Run every window instead of halting at the first STOP")
    ap.add_argument("--no-grasp-memory", action="store_true",
                    help="Disable the grasp COMMIT (do not retire the semantic check "
                         "after the right object is grasped; run the two-pass on every "
                         "window) — ablation")
    ap.add_argument("--grasp-close-margin", type=float, default=DEFAULT_GRASP_CLOSE_MARGIN,
                    help="Grasp-commit gate: commit only once the gripper aperture is "
                         "this far below the widest opening seen this run (fingers "
                         "closed on the object). 0 disables (VLM-only commit). "
                         f"Default: {DEFAULT_GRASP_CLOSE_MARGIN}")
    ap.add_argument("--out", default="vlm_monitor_test/monitor_test_results.json", help="Write per-episode results to this JSON file")
    ap.add_argument("--plot", default="vlm_monitor_test/confusion_matrix.png",
                    help="Write a confusion-matrix PNG here (full mode only). "
                         "Pass an empty string to disable.")
    args = ap.parse_args()

    # --- Ground truth ---
    if not os.path.isfile(args.worklist):
        print(f"[ERROR] Worklist not found: {args.worklist}")
        return
    ground_truth = load_ground_truth(args.worklist)
    print(f"Ground truth: {len(ground_truth)} high-quality episode(s) "
          f"(Success=TRUE, manual_label=non_oscillation, oscillation_class=normal) "
          f"from {args.worklist}")

    categories = [c.strip() for c in args.categories.split(",")] if args.categories else None
    episodes, missing = discover_episodes(args.root, ground_truth, categories,
                                          args.limit_per_category)
    if args.max_episodes is not None:
        episodes = episodes[:args.max_episodes]
    if not episodes:
        print(f"[ERROR] No usable episodes present under {args.root}"
              + (f" for categories {categories}" if categories else ""))
        if missing:
            print(f"  ({len(missing)} labelled-usable episode(s) are absent on disk)")
        return

    print(f"Using {len(episodes)} episode(s) present on disk across "
          f"{len({c for c, _, _, _ in episodes})} category(ies)"
          + (f"; {len(missing)} labelled episode(s) absent on disk" if missing else "")
          + f".\nMode: {'GATE-ONLY (no model)' if args.gate_only else 'FULL ACCURACY (two trials/episode)'}")
    print(f"Gate: start<{args.joint_start_thresh:g} OR still<{args.joint_still_thresh:g} "
          f"(pixel fallback <{args.motion_thresh:g}), window={args.window}\n")

    model = processor = prompts = None
    if not args.gate_only:
        from vlm_monitor_tune import load_model  # deferred: avoid model load in gate-only
        prompts = default_prompts()
        model, processor = load_model(args.model)

    results = []
    t0 = time.time()
    for k, (cat, name, ep_dir, obj) in enumerate(episodes, 1):
        rec = {"category": cat, "episode": name, "dir": ep_dir, "object_type": obj}

        # Skip episodes with no action log: the joint gate has nothing to run on, and
        # we do not want to silently fall back to the pixel gate for the whole test.
        if not os.path.isfile(os.path.join(ep_dir, "actions.csv")):
            rec.update({"error": "no action log (missing actions.csv)", "skipped_no_log": True})
            results.append(rec)
            print(f"[{k}/{len(episodes)}] {cat}/{name}: SKIP (no action log)")
            continue

        # In full mode we need a mismatch object for the wrong trial.
        wrong_obj = WRONG_OBJECT.get(obj, "").strip() if not args.gate_only else None
        if not args.gate_only and not wrong_obj:
            rec.update({"error": f"no WRONG_OBJECT mapping for object_type '{obj}'",
                        "skipped_no_mapping": True})
            results.append(rec)
            print(f"[{k}/{len(episodes)}] {cat}/{name}: SKIP (no WRONG_OBJECT['{obj}'])")
            continue

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                source = load_clip(ep_dir, view=args.view, wrist_view=args.wrist_view,
                                   wrist_rotate=args.wrist_rotate)
        except Exception as e:  # noqa: BLE001
            rec.update({"error": f"load failed: {e}"})
            results.append(rec)
            print(f"[{k}/{len(episodes)}] {cat}/{name}: LOAD ERROR ({e})")
            continue

        if len(source.clip) < args.window:
            rec.update({"error": f"only {len(source.clip)} frame(s) < window {args.window}"})
            results.append(rec)
            print(f"[{k}/{len(episodes)}] {cat}/{name}: too few frames ({len(source.clip)})")
            continue

        correct_task = resolve_task(source, cat)
        n_wrist = sum(1 for w in (source.wrist or []) if w is not None)
        rec.update({"task_correct": correct_task, "n_frames": len(source.clip),
                    "n_wrist": n_wrist, "joints_loaded": source.joints is not None})

        if args.gate_only:
            r = gate_only_episode(source, args.window, args.joint_start_thresh,
                                  args.joint_still_thresh, args.motion_thresh)
            rec.update(r)
            print(f"[{k}/{len(episodes)}] {cat}/{name}: joints={'Y' if rec['joints_loaded'] else 'N'} "
                  f"gate={r['gate_signal']} gated {r['n_gated']}/{r['n_windows']} "
                  f"-> VLM opens {r['first_vlm_window'] or '(never)'}")
        else:
            wrong_task = make_task(wrong_obj)
            rec["wrong_object"] = wrong_obj
            rec["task_wrong"] = wrong_task
            r = score_episode(model, processor, prompts, source,
                              correct_task, wrong_task, args)
            rec.update(r)
            cmark = "OK" if r["correct_pass"] else "FALSE-ALARM"
            wmark = "OK" if r["wrong_pass"] else "MISS"
            note = " (never opened)" if r["never_opened"] else ""
            print(f"[{k}/{len(episodes)}] {cat}/{name}: "
                  f"correct[{obj}]={'CONTINUE' if r['correct_pass'] else 'STOP'} {cmark}  |  "
                  f"wrong[{wrong_obj}]={'STOP' if r['wrong_pass'] else 'CONTINUE'} {wmark}{note}")
        results.append(rec)

    _print_summary(results, missing, args.gate_only)

    if not args.gate_only and args.plot:
        # Exclude fully-gated episodes (VLM never opened) from the matrix — they are
        # reported separately in the summary.
        scored = [r for r in results if "error" not in r and "correct" in r
                  and not r.get("never_opened")]
        plot_confusion_matrix(scored, args.plot)

    print(f"\nElapsed: {time.time() - t0:.1f}s")

    if args.out:
        fully_gated = [f"{r['category']}/{r['episode']}" for r in results
                       if r.get("never_opened")]
        Path(args.out).write_text(json.dumps(
            {"config": vars(args), "wrong_object_map": WRONG_OBJECT,
             "missing_on_disk": [f"{c}/{n}" for c, n in missing],
             "fully_gated_excluded": fully_gated,
             "results": results}, indent=2))
        print(f"Wrote {len(results)} episode records to {args.out}")


def _confusion_counts(ok: list[dict]) -> dict:
    """2x2 counts for the mismatch-detection task.

    Positive class = "wrong object" (arm is NOT on the sub-task's object -> should STOP).
    The correct trial is the negative class (arm IS on the object -> should CONTINUE).
      TP: wrong  trial, STOPped     (correct detection)
      FN: wrong  trial, CONTINUEd   (miss)
      FP: correct trial, STOPped    (false alarm)
      TN: correct trial, CONTINUEd  (correct continue)
    """
    tp = sum(1 for r in ok if r["wrong"]["stopped"])
    fn = sum(1 for r in ok if not r["wrong"]["stopped"])
    fp = sum(1 for r in ok if r["correct"]["stopped"])
    tn = sum(1 for r in ok if not r["correct"]["stopped"])
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn}


def plot_confusion_matrix(ok: list[dict], path: str) -> bool:
    """Save a 2x2 confusion-matrix PNG for the mismatch-detection task.

    Rows are the ACTUAL object condition (wrong vs. correct), columns the monitor's
    PREDICTED action (STOP vs. CONTINUE). Cells carry the count and its row share, on
    a single-hue sequential fill (darker = larger). Returns False (with a message) if
    matplotlib is unavailable or there is nothing to plot.
    """
    if not ok:
        print("[plot] nothing to plot (no scored episodes).")
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:  # noqa: BLE001
        print(f"[plot] matplotlib unavailable ({e}); skipping confusion-matrix plot.")
        return False

    c = _confusion_counts(ok)
    # Layout: rows = actual (wrong, correct), cols = predicted (STOP, CONTINUE).
    mat = np.array([[c["TP"], c["FN"]],
                    [c["FP"], c["TN"]]], dtype=float)
    row_tot = mat.sum(axis=1, keepdims=True)
    frac = np.divide(mat, row_tot, out=np.zeros_like(mat), where=row_tot > 0)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    # Sequential single hue: magnitude of the count. Normalised per its own max so
    # the largest cell anchors the dark end.
    vmax = mat.max() if mat.max() > 0 else 1.0
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax)

    ax.set_xticks([0, 1], ["STOP", "CONTINUE"])
    ax.set_yticks([0, 1], ["wrong object\n(should STOP)", "correct object\n(should CONTINUE)"])
    ax.set_xlabel("monitor action (predicted)")
    ax.set_ylabel("actual object condition")
    n = len(ok)
    ax.set_title(f"Semantic monitor confusion matrix\n{n} episode(s), {2*n} trials")

    labels = [["TP", "FN"], ["FP", "TN"]]  # cell role, direct-labelled (not colour-alone)
    for i in range(2):
        for j in range(2):
            dark = mat[i, j] > 0.5 * vmax
            ax.text(j, i, f"{labels[i][j]}\n{int(mat[i, j])}  ({100*frac[i, j]:.0f}%)",
                    ha="center", va="center",
                    color="white" if dark else "#1f2933", fontsize=11,
                    fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("episode count")
    # Recessive frame: drop the top/right spines, keep the grid off.
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    # Footnote with the headline rates so the figure stands alone.
    recall = c["TP"] / (c["TP"] + c["FN"]) if (c["TP"] + c["FN"]) else 0.0
    spec = c["TN"] / (c["TN"] + c["FP"]) if (c["TN"] + c["FP"]) else 0.0
    fig.text(0.5, 0.01,
             f"recall (wrong->STOP) {100*recall:.0f}%   |   "
             f"specificity (correct->CONTINUE) {100*spec:.0f}%",
             ha="center", va="bottom", fontsize=9, color="#52606d")

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    try:
        fig.savefig(path, dpi=150)
    except Exception as e:  # noqa: BLE001
        print(f"[plot] could not write {path}: {e}")
        plt.close(fig)
        return False
    plt.close(fig)
    print(f"Wrote confusion-matrix plot to {path}")
    return True


def _print_summary(results: list[dict], missing: list[tuple[str, str]],
                   gate_only: bool) -> None:
    ok = [r for r in results if "error" not in r]
    no_log = [r for r in results if r.get("skipped_no_log")]
    no_map = [r for r in results if r.get("skipped_no_mapping")]
    other_errs = [r for r in results
                  if "error" in r and not r.get("skipped_no_log")
                  and not r.get("skipped_no_mapping")]
    print(f"\n{'#'*64}\nSUMMARY — {len(ok)} episode(s) scored, "
          f"{len(no_log)} no-log, {len(no_map)} no-mapping, "
          f"{len(other_errs)} other skips/errors, {len(missing)} absent on disk"
          f"\n{'#'*64}")
    if no_map:
        types = sorted({r["object_type"] for r in no_map})
        print(f"  Fill WRONG_OBJECT for: {', '.join(types)}")
    if not ok:
        return
    no_joints = [r for r in ok if not r.get("joints_loaded")]
    if no_joints:
        print(f"  {len(no_joints)} scored episode(s) had a CSV but no usable joint columns")

    cats = sorted({r["category"] for r in ok})
    if gate_only:
        print(f"\n  {'category':<14}{'eps':>5}{'all-gated':>11}{'avg gated%':>12}")
        for c in cats:
            rs = [r for r in ok if r["category"] == c]
            allg = sum(1 for r in rs if r.get("all_gated"))
            pct = 100 * sum(r["n_gated"] for r in rs) / max(1, sum(r["n_windows"] for r in rs))
            print(f"  {c:<14}{len(rs):>5}{allg:>11}{pct:>11.0f}%")
        total_allg = sum(1 for r in ok if r.get("all_gated"))
        print(f"\n  {total_allg}/{len(ok)} episode(s) fully gated (VLM never opened).")
        return

    # --- Accuracy summary (two trials per episode) ---
    # Episodes whose gate never opened (fully gated across the sweep) are EXCLUDED
    # from the confusion matrix and accuracy: the VLM was never asked, so they say
    # nothing about the monitor's decisions. They are listed separately below.
    never_eps = [r for r in ok if r.get("never_opened")]
    opened = [r for r in ok if not r.get("never_opened")]

    if never_eps:
        print(f"\n  EXCLUDED — {len(never_eps)} episode(s) fully gated (VLM never opened; "
              "not scored):")
        for r in never_eps:
            print(f"    {r['category']}/{r['episode']}  (object: {r.get('object_type','?')})")

    if not opened:
        print("\n  No episodes opened the VLM — nothing to score.")
        return

    print(f"\n  correct trial: monitor SHOULD continue (STOP = false alarm)")
    print(f"  wrong trial:   monitor SHOULD stop     (no STOP = miss)")
    print(f"  (scored on the {len(opened)} episode(s) whose gate opened)\n")
    print(f"  {'category':<12}{'eps':>5}{'specificity':>13}{'recall':>9}")
    for c in cats:
        rs = [r for r in opened if r["category"] == c]
        if not rs:
            continue
        spec = sum(1 for r in rs if r["correct_pass"]) / len(rs)
        rec = sum(1 for r in rs if r["wrong_pass"]) / len(rs)
        print(f"  {c:<12}{len(rs):>5}{100*spec:>12.0f}%{100*rec:>8.0f}%")

    n = len(opened)
    spec = sum(1 for r in opened if r["correct_pass"]) / n
    recall = sum(1 for r in opened if r["wrong_pass"]) / n
    false_alarms = sum(1 for r in opened if not r["correct_pass"])
    misses = sum(1 for r in opened if not r["wrong_pass"])
    parse_fail = sum(r["correct"].get("n_parse_fail", 0) + r["wrong"].get("n_parse_fail", 0)
                     for r in opened)
    balanced = (spec + recall) / 2
    trial_acc = (sum(1 for r in opened if r["correct_pass"])
                 + sum(1 for r in opened if r["wrong_pass"])) / (2 * n)
    print(f"\n  TOTAL over {n} scored episode(s) ({2*n} trials; "
          f"{len(never_eps)} fully-gated excluded):")
    print(f"    specificity (correct-object, no false STOP): {100*spec:5.1f}%  "
          f"({false_alarms} false alarm(s))")
    print(f"    recall      (wrong-object, correctly STOP):   {100*recall:5.1f}%  "
          f"({misses} miss(es))")
    print(f"    balanced accuracy (mean of the two):          {100*balanced:5.1f}%")
    print(f"    per-trial accuracy ({2*n} trials):            {100*trial_acc:5.1f}%")
    if parse_fail:
        print(f"    note: {parse_fail} window(s) failed JSON parsing across all trials.")


if __name__ == "__main__":
    main()
