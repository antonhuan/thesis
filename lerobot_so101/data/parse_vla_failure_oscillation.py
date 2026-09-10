#!/usr/bin/env python3
"""Scan ``vla_failure_test/`` for episodes with oscillation behaviour.

Layout scanned::

    vla_failure_test/<object>/<NNN_object>/actions.csv

Each ``actions.csv`` is the split (vla_failure_test) format: one row per
executed action with ``in_<joint>.pos`` (measured) and ``act_<joint>.pos``
(commanded) columns for the six SO-101 joints, plus ``input_elapsed`` giving
the wall-clock time (seconds) of each VLM inference chunk.

Oscillation is defined exactly as in ``parse_pouch_episodes.py``: a *major
reversal* is a direction change that occurs only after at least
``--reversal-threshold`` degrees (default 15°) of travel since the previous
reversal.  Rates are measured **per timestep** (per executed action), not per
real second.  Classification is evaluated over the analysed joints
(shoulder_pan, shoulder_lift, elbow_flex, wrist_flex -- gripper and wrist_roll
are excluded): an episode is ``oscillation`` when *any* analysed joint sweeps a
wide range (> ``--min-range``) while reversing often (>= ``--min-rev-per-step``
major reversals per timestep).

Separately, a **manual-review flag** captures two velocity-based signatures
with an inclusive OR: an episode is flagged for review if any analysed joint
has high mean velocity (>= ``--vel-threshold`` deg/step) OR many high-velocity
sign flips (>= ``--fast-rev-threshold`` reversals among fast steps, where a
"fast" step moves >= ``--fast-step`` deg).  This is meant as a high-recall net
to hand-check, not a precise classifier.

Outputs (written to ``vla_failure_analysis/``):

  * oscillation_manifest.csv  one row per episode with per-joint reversal
                               counts, ranges, rev/step, mean speed, fast-rev
                               counts, the class label, and the review flag
  * oscillating_episodes.txt   episode ids classified as oscillating (reversal)
  * review_episodes.txt        episode ids flagged for manual review (velocity
                               OR sign flips), tab-separated with the reason

Run from ``lerobot_so101/data/``::

    python parse_vla_failure_oscillation.py                 # analyse
    python parse_vla_failure_oscillation.py --source measured
    python parse_vla_failure_oscillation.py --plots         # + diagnostic plots
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Joints used for oscillation analysis. gripper and wrist_roll are excluded:
# the gripper cycles open/close by design and wrist_roll barely moves, so
# neither reflects the arm-oscillation failure mode.
JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
]

TEST_DIR = Path("vla_failure_test")
OUT_DIR = Path("vla_failure_analysis")

REVERSAL_THRESHOLD_DEG = 15.0

# A step counts as "fast" when the commanded joint moves at least this many
# degrees in one action. "Big sign flips" counts how often the direction of
# fast motion reverses (the red<->blue alternation in the velocity plots).
FAST_STEP_DEG = 2.0


def fast_sign_flips(positions: np.ndarray, fast_step_deg: float) -> int:
    """Count velocity sign reversals among 'fast' steps (|delta| >= fast_step_deg).

    This isolates high-velocity back-and-forth motion: near-zero jitter is
    ignored, and only direction changes between fast moves are counted.
    """
    if len(positions) < 2:
        return 0
    v = np.diff(positions)
    fast_signs = np.sign(v[np.abs(v) >= fast_step_deg])
    if len(fast_signs) < 2:
        return 0
    return int((np.diff(fast_signs) != 0).sum())


def find_major_reversals(positions: np.ndarray, threshold_deg: float) -> list[int]:
    """Direction changes only after >= threshold_deg travel since last reversal.

    Identical logic to parse_pouch_episodes.find_major_reversals /
    plot_oscillation_diagnostics.find_major_reversals.
    """
    if len(positions) < 3:
        return []
    reversals: list[int] = []
    last_rev_pos = positions[0]
    last_direction = None
    for i in range(1, len(positions)):
        delta = positions[i] - positions[i - 1]
        if abs(delta) < 0.01:
            continue
        current_dir = 1 if delta > 0 else -1
        if last_direction is not None and current_dir != last_direction:
            if abs(positions[i] - last_rev_pos) >= threshold_deg:
                reversals.append(i)
                last_rev_pos = positions[i]
        last_direction = current_dir
    return reversals


def load_actions_csv(path: Path, source: str) -> tuple[dict[str, np.ndarray], np.ndarray, str]:
    """Return ({joint: position array}, elapsed_time array, task) from actions.csv.

    source: 'commanded' -> act_<joint>.pos, 'measured' -> in_<joint>.pos.
    """
    prefix = "act_" if source == "commanded" else "in_"
    cols: dict[str, list[float]] = {j: [] for j in JOINTS}
    elapsed: list[float] = []
    task = ""
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not task:
                task = row.get("task", "") or ""
            for j in JOINTS:
                val = row.get(f"{prefix}{j}.pos", "")
                cols[j].append(float(val) if val not in ("", None) else np.nan)
            e = row.get("input_elapsed", "")
            elapsed.append(float(e) if e not in ("", None) else np.nan)
    return {j: np.array(v) for j, v in cols.items()}, np.array(elapsed), task


def analyse_episode(positions: dict[str, np.ndarray], elapsed: np.ndarray,
                    threshold_deg: float, min_rev_per_step: float, min_range: float,
                    fast_step_deg: float, vel_threshold: float,
                    fast_rev_threshold: int) -> dict:
    """Compute per-joint oscillation metrics and a class label for one episode.

    Two independent views are produced:

    * ``oscillation_class`` -- the timestep-based major-reversal metric: an
      episode oscillates if any analysed joint sweeps > min_range degrees and
      reverses >= min_rev_per_step reversals per timestep.
    * ``review_flag`` -- an inclusive OR net for manual review: True if any
      joint has high mean velocity (>= vel_threshold deg/step) OR many
      high-velocity sign flips (>= fast_rev_threshold).
    """
    n = len(next(iter(positions.values())))
    n_steps = n - 1  # number of step-to-step intervals
    # Real elapsed time is kept for reference only; the rate uses timesteps.
    finite = elapsed[np.isfinite(elapsed)]
    duration = float(finite[-1] - finite[0]) if len(finite) >= 2 else 0.0

    metrics: dict = {"num_actions": n, "duration_s": round(duration, 3)}

    ranges: dict[str, float] = {}
    rev_per_step: dict[str, float] = {}
    rev_counts: dict[str, int] = {}
    mean_speeds: dict[str, float] = {}
    fast_revs: dict[str, int] = {}
    oscillating_joints: list[str] = []

    for j in JOINTS:
        pos = positions[j]
        pos = pos[np.isfinite(pos)]
        if len(pos) < 2:
            rng, revs, mean_speed, fflip = 0.0, [], 0.0, 0
        else:
            rng = float(np.nanmax(pos) - np.nanmin(pos))
            revs = find_major_reversals(pos, threshold_deg)
            mean_speed = float(np.abs(np.diff(pos)).mean())
            fflip = fast_sign_flips(pos, fast_step_deg)
        rps = len(revs) / n_steps if n_steps > 0 else 0.0
        ranges[j] = round(rng, 1)
        rev_per_step[j] = round(rps, 4)
        rev_counts[j] = len(revs)
        mean_speeds[j] = round(mean_speed, 3)
        fast_revs[j] = fflip
        metrics[f"{j}_major_rev"] = len(revs)
        metrics[f"{j}_range"] = round(rng, 1)
        metrics[f"{j}_rev_per_step"] = round(rps, 4)
        metrics[f"{j}_mean_speed"] = round(mean_speed, 3)
        metrics[f"{j}_fast_rev"] = fflip
        if rng > min_range and rps >= min_rev_per_step:
            oscillating_joints.append(j)

    metrics["total_major_rev"] = sum(rev_counts.values())
    metrics["max_range"] = round(max(ranges.values()), 1)
    metrics["max_rev_per_step"] = round(max(rev_per_step.values()), 4)
    metrics["oscillating_joints"] = ";".join(oscillating_joints)
    metrics["n_oscillating_joints"] = len(oscillating_joints)

    if oscillating_joints:
        cls = "oscillation"
    elif max(ranges.values()) < 10:
        cls = "zero_movement"
    else:
        cls = "normal"
    metrics["oscillation_class"] = cls

    # --- velocity / sign-flip review flag (inclusive OR) ---
    max_mean_speed = max(mean_speeds.values())
    max_fast_rev = max(fast_revs.values())
    metrics["max_mean_speed"] = round(max_mean_speed, 3)
    metrics["max_fast_rev"] = max_fast_rev
    high_velocity = max_mean_speed >= vel_threshold
    big_sign_flips = max_fast_rev >= fast_rev_threshold
    reasons = []
    if high_velocity:
        reasons.append("high_velocity")
    if big_sign_flips:
        reasons.append("big_sign_flips")
    metrics["high_velocity"] = high_velocity
    metrics["big_sign_flips"] = big_sign_flips
    metrics["review_flag"] = bool(reasons)
    metrics["review_reason"] = ";".join(reasons)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan vla_failure_test episodes for oscillation behaviour.")
    parser.add_argument("--test-dir", default=str(TEST_DIR),
                        help=f"Root of episode tree (default: {TEST_DIR})")
    parser.add_argument("--out-dir", default=str(OUT_DIR),
                        help=f"Output directory (default: {OUT_DIR})")
    parser.add_argument("--source", choices=["commanded", "measured"], default="commanded",
                        help="Use commanded (act_) or measured (in_) positions (default: commanded)")
    parser.add_argument("--reversal-threshold", type=float, default=REVERSAL_THRESHOLD_DEG,
                        help="Major reversal threshold in degrees (default: 15.0)")
    parser.add_argument("--min-rev-per-step", type=float, default=0.02,
                        help="Min reversals per timestep (any joint) to call oscillation "
                             "(default: 0.02)")
    parser.add_argument("--min-range", type=float, default=50.0,
                        help="Min joint range (deg) to call oscillation (default: 50.0)")
    parser.add_argument("--fast-step", type=float, default=FAST_STEP_DEG,
                        help="Per-step move (deg) that counts as a 'fast' step for "
                             "sign-flip counting (default: 2.0)")
    parser.add_argument("--vel-threshold", type=float, default=2.0,
                        help="Review flag: min mean speed (deg/step, any joint) that "
                             "counts as high velocity (default: 2.0)")
    parser.add_argument("--fast-rev-threshold", type=int, default=23,
                        help="Review flag: min high-velocity sign flips (any joint) that "
                             "count as big sign flips (default: 23)")
    parser.add_argument("--plots", action="store_true",
                        help="Generate diagnostic plots for oscillating episodes "
                             "(via plot_oscillation_diagnostics.py)")
    args = parser.parse_args()

    test_dir = Path(args.test_dir)
    out_dir = Path(args.out_dir)
    if not test_dir.is_dir():
        print(f"ERROR: not a directory: {test_dir}")
        sys.exit(1)

    csv_paths = sorted(test_dir.glob("*/*/actions.csv"))
    if not csv_paths:
        print(f"ERROR: no */*/actions.csv found under {test_dir}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "plots"
    script_dir = Path(__file__).resolve().parent

    manifest_fields = ["episode_id", "object_type", "episode_dir", "task",
                       "num_actions", "duration_s", "source"]
    for j in JOINTS:
        manifest_fields += [f"{j}_major_rev", f"{j}_range", f"{j}_rev_per_step",
                            f"{j}_mean_speed", f"{j}_fast_rev"]
    manifest_fields += ["total_major_rev", "max_range", "max_rev_per_step",
                        "oscillating_joints", "n_oscillating_joints", "oscillation_class",
                        "max_mean_speed", "max_fast_rev", "high_velocity",
                        "big_sign_flips", "review_flag", "review_reason"]

    rows: list[dict] = []
    warnings: list[str] = []
    oscillating: list[str] = []
    review: list[str] = []

    for csv_path in csv_paths:
        ep_dir = csv_path.parent
        object_type = ep_dir.parent.name
        episode_dir_name = ep_dir.name
        episode_id = f"{object_type}__{episode_dir_name}"

        try:
            positions, elapsed, task = load_actions_csv(csv_path, args.source)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{episode_id}: failed to read actions.csv ({exc})")
            continue
        if len(next(iter(positions.values()))) == 0:
            warnings.append(f"{episode_id}: empty actions.csv")
            continue

        m = analyse_episode(positions, elapsed, args.reversal_threshold,
                            args.min_rev_per_step, args.min_range,
                            args.fast_step, args.vel_threshold, args.fast_rev_threshold)

        row = {"episode_id": episode_id, "object_type": object_type,
               "episode_dir": episode_dir_name, "task": task, "source": args.source}
        row.update(m)
        rows.append(row)

        if m["oscillation_class"] == "oscillation":
            oscillating.append(episode_id)
        if m["review_flag"]:
            review.append(episode_id)
            if args.plots:
                plot_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [sys.executable, str(script_dir / "plot_oscillation_diagnostics.py"),
                     str(csv_path), "-o", str(plot_dir), "-t", episode_id,
                     "--threshold", str(args.reversal_threshold)],
                    check=False,
                )

    manifest_path = out_dir / "oscillation_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=manifest_fields)
        w.writeheader()
        w.writerows(rows)

    osc_list_path = out_dir / "oscillating_episodes.txt"
    osc_list_path.write_text("\n".join(oscillating) + ("\n" if oscillating else ""))

    # Review list: high-velocity OR big sign flips, with the reason per episode.
    review_path = out_dir / "review_episodes.txt"
    review_lines = []
    for ep_id in review:
        r = next(x for x in rows if x["episode_id"] == ep_id)
        review_lines.append(f"{ep_id}\t{r['review_reason']}")
    review_path.write_text("\n".join(review_lines) + ("\n" if review_lines else ""))

    # --- summary ---
    by_class = Counter(r["oscillation_class"] for r in rows)
    print(f"Scanned {len(rows)} episodes under {test_dir} (source={args.source}, "
          f"reversal-threshold={args.reversal_threshold}°).")
    print(f"Classes: {', '.join(f'{k}={v}' for k, v in sorted(by_class.items()))}")
    print(f"\n{len(oscillating)} episode(s) with OSCILLATION behaviour "
          f"(any joint: range > {args.min_range}°, >= {args.min_rev_per_step} rev/step):")
    for ep_id in oscillating:
        r = next(x for x in rows if x["episode_id"] == ep_id)
        print(f"  {ep_id:28s} max_range={r['max_range']:6.1f}°  "
              f"max_rev/step={r['max_rev_per_step']:.4f}  joints=[{r['oscillating_joints']}]")

    n_hv = sum(1 for r in rows if r["high_velocity"])
    n_bf = sum(1 for r in rows if r["big_sign_flips"])
    print(f"\n{len(review)} episode(s) FLAGGED FOR MANUAL REVIEW "
          f"(mean speed >= {args.vel_threshold} deg/step OR "
          f">= {args.fast_rev_threshold} fast sign flips) "
          f"[high_velocity={n_hv}, big_sign_flips={n_bf}]:")
    for ep_id in review:
        r = next(x for x in rows if x["episode_id"] == ep_id)
        print(f"  {ep_id:28s} mean_speed={r['max_mean_speed']:5.2f}  "
              f"fast_rev={r['max_fast_rev']:3d}  reason=[{r['review_reason']}]")
    print(f"\nWrote {manifest_path}")
    print(f"Wrote {osc_list_path}")
    print(f"Wrote {review_path}")
    if args.plots:
        print(f"Plots: {len(review)} review episodes × 3 in {plot_dir}/")
    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for wmsg in warnings:
            print(f"  - {wmsg}")


if __name__ == "__main__":
    main()
