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
reversal.  Classification is evaluated over **every joint**: an episode is
``oscillation`` when *any* joint sweeps a wide range (> ``--min-range``) while
reversing often (>= ``--min-rev-per-s`` major reversals per second).

Outputs (written to ``vla_failure_analysis/``):

  * oscillation_manifest.csv  one row per episode with per-joint reversal
                               counts, ranges, rev/s, the oscillating joints,
                               and the class label
  * oscillating_episodes.txt   just the episode ids classified as oscillating

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

JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

TEST_DIR = Path("vla_failure_test")
OUT_DIR = Path("vla_failure_analysis")

REVERSAL_THRESHOLD_DEG = 15.0


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
                    threshold_deg: float, min_rev_per_s: float, min_range: float) -> dict:
    """Compute per-joint oscillation metrics and a class label for one episode.

    Classification considers every joint: the episode oscillates if any single
    joint both sweeps > min_range degrees and reverses >= min_rev_per_s / sec.
    """
    n = len(next(iter(positions.values())))
    # Duration from the elapsed-seconds column (per-chunk wall clock).
    finite = elapsed[np.isfinite(elapsed)]
    duration = float(finite[-1] - finite[0]) if len(finite) >= 2 else 0.0

    metrics: dict = {"num_actions": n, "duration_s": round(duration, 3)}

    ranges: dict[str, float] = {}
    rev_per_s: dict[str, float] = {}
    rev_counts: dict[str, int] = {}
    oscillating_joints: list[str] = []

    for j in JOINTS:
        pos = positions[j]
        pos = pos[np.isfinite(pos)]
        if len(pos) < 2:
            rng, revs = 0.0, []
        else:
            rng = float(np.nanmax(pos) - np.nanmin(pos))
            revs = find_major_reversals(pos, threshold_deg)
        rps = len(revs) / duration if duration > 0 else 0.0
        ranges[j] = round(rng, 1)
        rev_per_s[j] = round(rps, 3)
        rev_counts[j] = len(revs)
        metrics[f"{j}_major_rev"] = len(revs)
        metrics[f"{j}_range"] = round(rng, 1)
        metrics[f"{j}_rev_per_s"] = round(rps, 3)
        if rng > min_range and rps >= min_rev_per_s:
            oscillating_joints.append(j)

    metrics["total_major_rev"] = sum(rev_counts.values())
    metrics["max_range"] = round(max(ranges.values()), 1)
    metrics["max_rev_per_s"] = round(max(rev_per_s.values()), 3)
    metrics["oscillating_joints"] = ";".join(oscillating_joints)
    metrics["n_oscillating_joints"] = len(oscillating_joints)

    if oscillating_joints:
        cls = "oscillation"
    elif max(ranges.values()) < 10:
        cls = "zero_movement"
    else:
        cls = "normal"
    metrics["oscillation_class"] = cls
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
    parser.add_argument("--min-rev-per-s", type=float, default=0.5,
                        help="Min reversals/sec (any joint) to call oscillation (default: 0.5)")
    parser.add_argument("--min-range", type=float, default=50.0,
                        help="Min joint range (deg) to call oscillation (default: 50.0)")
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
        manifest_fields += [f"{j}_major_rev", f"{j}_range", f"{j}_rev_per_s"]
    manifest_fields += ["total_major_rev", "max_range", "max_rev_per_s",
                        "oscillating_joints", "n_oscillating_joints", "oscillation_class"]

    rows: list[dict] = []
    warnings: list[str] = []
    oscillating: list[str] = []

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
                            args.min_rev_per_s, args.min_range)

        row = {"episode_id": episode_id, "object_type": object_type,
               "episode_dir": episode_dir_name, "task": task, "source": args.source}
        row.update(m)
        rows.append(row)

        if m["oscillation_class"] == "oscillation":
            oscillating.append(episode_id)
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

    # --- summary ---
    by_class = Counter(r["oscillation_class"] for r in rows)
    print(f"Scanned {len(rows)} episodes under {test_dir} (source={args.source}, "
          f"reversal-threshold={args.reversal_threshold}°).")
    print(f"Classes: {', '.join(f'{k}={v}' for k, v in sorted(by_class.items()))}")
    print(f"\n{len(oscillating)} episode(s) with OSCILLATION behaviour "
          f"(any joint: range > {args.min_range}°, >= {args.min_rev_per_s} rev/s):")
    for ep_id in oscillating:
        r = next(x for x in rows if x["episode_id"] == ep_id)
        print(f"  {ep_id:28s} max_range={r['max_range']:6.1f}°  "
              f"max_rev/s={r['max_rev_per_s']:.3f}  joints=[{r['oscillating_joints']}]")
    print(f"\nWrote {manifest_path}")
    print(f"Wrote {osc_list_path}")
    if args.plots:
        print(f"Plots: {len(oscillating)} episodes × 3 in {plot_dir}/")
    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for wmsg in warnings:
            print(f"  - {wmsg}")


if __name__ == "__main__":
    main()
