#!/usr/bin/env python3
"""Plot per-joint step-velocity std as a single-feature oscillation discriminator.

This is an *exploratory* companion to ``plot_oscillation_separation.py``. Where
that script visualises the shipped review flag (the mean AND peak conjunction),
this one asks whether a single scalar can do the same job.

The candidate feature is the standard deviation of the per-step commanded
velocity (``std(diff(pos))``) of each analysed joint. Because oscillation swings
back and forth, its mean velocity is ~0, so this std is essentially the RMS
step speed -- and RMS folds "sustained fast motion" (mean) and "large single-step
excursions" (peak) into one number. That is why it reproduces the conjunction:
it is strongly correlated with max_mean_speed (~0.96) but not a pure rescaling of
it (the std/mean ratio, i.e. the crest factor, varies ~1.2-5.2 across episodes),
so it also carries the peak information the mean gate lacks.

On the hand-labelled test set the MAX-across-joints std separates the two classes
with no overlap: oscillation bottoms out at ~11.1 (banana__110, shoulder_lift)
while non-oscillation tops out at ~9.8 (fail__114), a ~14% margin around the
``--threshold`` default of 10.45. The separation lives almost entirely in
shoulder_lift, with elbow_flex as a redundant second witness; shoulder_pan and
wrist_flex do not discriminate.

NOTE: with only 7 positive episodes a clean 1-D split is fragile and partly
expected by chance. Treat this as a diagnostic, not a replacement for the shipped
conjunction, unless the margin holds as the labelled set grows.

Ground-truth labels are read from the worklist CSV (``manual_label``); the
per-joint std is recomputed from each episode's ``actions.csv`` under
``--test-dir`` (the worklist does not store it). Joints and CSV loading are
shared with ``parse_vla_failure_oscillation.py`` so the definitions stay in sync.

Usage (run from ``lerobot_so101/data/``)::

    python plot_velocity_std_separation.py
    python plot_velocity_std_separation.py --threshold 10.45
    python plot_velocity_std_separation.py --test-dir vla_failure_test -o vla_failure_analysis2
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)

from parse_vla_failure_oscillation import JOINTS, load_actions_csv

DEFAULT_WORKLIST = Path("vla_failure_analysis2/labeling_worklist.csv")
DEFAULT_TEST_DIR = Path("vla_failure_test")
DEFAULT_OUT_DIR = Path("vla_failure_analysis2")

# Default single-feature gate: midpoint of the observed class gap (9.78 -> 11.12)
# on the 227-episode labelled set. Not a shipped threshold -- see module docstring.
STD_THRESHOLD = 10.45

OSC_COLOR = "#c0392b"
NON_COLOR = "#2a78d6"


def load_labels(path: Path) -> dict[str, str]:
    """Return {episode_id: manual_label} for every labelled worklist row."""
    labels: dict[str, str] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            eid = (row.get("episode_id") or "").strip()
            lab = (row.get("manual_label") or "").strip()
            if eid and lab:
                labels[eid] = lab
    return labels


def joint_velocity_std(csv_path: Path, source: str) -> dict[str, float]:
    """Return {joint: std of per-step commanded velocity} for one episode."""
    positions, _elapsed, _task = load_actions_csv(csv_path, source)
    out: dict[str, float] = {}
    for j in JOINTS:
        p = positions[j]
        p = p[np.isfinite(p)]
        out[j] = float(np.diff(p).std()) if len(p) >= 4 else 0.0
    return out


def collect(test_dir: Path, labels: dict[str, str], source: str):
    """Gather per-joint std lists split by class, plus the max-across-joints value.

    Returns (per_joint, maxvals) where per_joint[joint][cls] and maxvals[cls] are
    lists of floats and cls is 'osc' / 'non'.
    """
    per_joint = {j: {"osc": [], "non": []} for j in JOINTS}
    maxvals = {"osc": [], "non": []}
    for csv_path in sorted(test_dir.glob("*/*/actions.csv")):
        eid = f"{csv_path.parent.parent.name}__{csv_path.parent.name}"
        if eid not in labels:
            continue
        cls = "osc" if labels[eid] == "oscillation" else "non"
        stds = joint_velocity_std(csv_path, source)
        for j in JOINTS:
            per_joint[j][cls].append(stds[j])
        maxvals[cls].append(max(stds.values()))
    return per_joint, maxvals


def _strip(ax, xs_center, ys, cls, rng, jitter=0.06):
    xs = xs_center + rng.uniform(-jitter, jitter, size=len(ys))
    ax.scatter(xs, ys,
               s=(60 if cls == "osc" else 24),
               c=(OSC_COLOR if cls == "osc" else NON_COLOR),
               alpha=(0.9 if cls == "osc" else 0.4),
               edgecolor=("#7a1c14" if cls == "osc" else "none"),
               linewidth=0.8, zorder=(3 if cls == "osc" else 2))


def make_plot(per_joint, maxvals, threshold: float, out_path: Path) -> None:
    rng = np.random.default_rng(0)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [3, 1]})

    # Panel 1: per-joint strip plot.
    for i, j in enumerate(JOINTS):
        _strip(ax1, i - 0.12, per_joint[j]["non"], "non", rng)
        _strip(ax1, i + 0.12, per_joint[j]["osc"], "osc", rng)
    ax1.axhline(threshold, color="#555", linestyle="--", linewidth=1.3, zorder=1)
    ax1.text(len(JOINTS) - 0.5, threshold, f" threshold = {threshold:g}",
             va="bottom", ha="right", color="#555", fontsize=9)
    ax1.set_xticks(range(len(JOINTS)))
    ax1.set_xticklabels(JOINTS, rotation=15)
    ax1.set_ylabel("per-step velocity std (deg/step)")
    n_total = len(maxvals["osc"]) + len(maxvals["non"])
    ax1.set_title(f"Step-velocity std per joint ({n_total} labelled episodes)")
    ax1.grid(True, axis="y", color="0.9", zorder=0)

    # Panel 2: max across joints -- the actual single-feature discriminator.
    _strip(ax2, -0.12, maxvals["non"], "non", rng, jitter=0.05)
    _strip(ax2, 0.12, maxvals["osc"], "osc", rng, jitter=0.05)
    ax2.axhline(threshold, color="#555", linestyle="--", linewidth=1.3, zorder=1)
    ax2.axhspan(threshold, ax2.get_ylim()[1], color=OSC_COLOR, alpha=0.06, zorder=0)
    ax2.set_xticks([])
    ax2.set_xlim(-0.35, 0.35)
    ax2.set_title("max across joints\n(candidate feature)")
    ax2.grid(True, axis="y", color="0.9", zorder=0)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=OSC_COLOR,
               markeredgecolor="#7a1c14", markersize=9,
               label=f"oscillation ({len(maxvals['osc'])})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=NON_COLOR,
               markeredgecolor="none", markersize=7,
               label=f"non-oscillation ({len(maxvals['non'])})"),
    ]
    ax1.legend(handles=handles, loc="upper left", framealpha=0.9)

    osc_min = min(maxvals["osc"]) if maxvals["osc"] else float("nan")
    non_max = max(maxvals["non"]) if maxvals["non"] else float("nan")
    fig.suptitle("Oscillation separation by step-velocity std  "
                 f"(gap: {non_max:.2f} -> {osc_min:.2f}, threshold {threshold:g})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-joint step-velocity std as an oscillation discriminator.")
    parser.add_argument("--worklist", default=str(DEFAULT_WORKLIST),
                        help=f"Labelled worklist CSV (default: {DEFAULT_WORKLIST})")
    parser.add_argument("--test-dir", default=str(DEFAULT_TEST_DIR),
                        help=f"Root of episode tree (default: {DEFAULT_TEST_DIR})")
    parser.add_argument("-o", "--out-dir", default=str(DEFAULT_OUT_DIR),
                        help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--source", choices=["commanded", "measured"], default="commanded",
                        help="Use commanded (act_) or measured (in_) positions (default: commanded)")
    parser.add_argument("--threshold", type=float, default=STD_THRESHOLD,
                        help=f"Std gate to draw (default: {STD_THRESHOLD})")
    args = parser.parse_args()

    worklist = Path(args.worklist)
    test_dir = Path(args.test_dir)
    if not worklist.is_file():
        print(f"ERROR: worklist not found: {worklist}")
        sys.exit(1)
    if not test_dir.is_dir():
        print(f"ERROR: not a directory: {test_dir}")
        sys.exit(1)

    labels = load_labels(worklist)
    if not labels:
        print(f"ERROR: no labelled rows in {worklist}")
        sys.exit(1)

    per_joint, maxvals = collect(test_dir, labels, args.source)
    if not maxvals["osc"] and not maxvals["non"]:
        print(f"ERROR: no labelled episodes found under {test_dir}")
        sys.exit(1)

    out_path = Path(args.out_dir) / "velocity_std_separation.png"
    make_plot(per_joint, maxvals, args.threshold, out_path)

    # Report the split under the drawn threshold (max-across-joints feature).
    def flagged(v: float) -> bool:
        return v >= args.threshold

    tp = sum(flagged(v) for v in maxvals["osc"])
    fp = sum(flagged(v) for v in maxvals["non"])
    osc_min = min(maxvals["osc"]) if maxvals["osc"] else float("nan")
    non_max = max(maxvals["non"]) if maxvals["non"] else float("nan")
    print(f"Labelled: {len(maxvals['osc'])} oscillation, {len(maxvals['non'])} "
          f"non-oscillation -- {len(maxvals['osc']) + len(maxvals['non'])} total.")
    print(f"Feature: max per-joint step-velocity std. Gate: std >= {args.threshold:g}")
    print(f"  flagged oscillation (TP)     = {tp}/{len(maxvals['osc'])}")
    print(f"  flagged non-oscillation (FP) = {fp}/{len(maxvals['non'])}")
    print(f"  class gap: non_max={non_max:.3f}  ->  osc_min={osc_min:.3f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
