#!/usr/bin/env python3
"""Plot the review-flag threshold separation on the hand-labelled test set.

Reads ``vla_failure_analysis/labeling_worklist.csv`` (the manually labelled
episodes, one row each with ``max_mean_speed``, ``max_peak_speed`` and the
``manual_label`` ground truth) and draws a scatter of max mean per-step
velocity (x) against max peak per-step velocity (y), coloured by manual label.

The two review-flag gates are overlaid as dashed lines and the conjunction
(mean >= VEL_THRESHOLD AND peak >= PEAK_THRESHOLD) is shaded: this is the
region in which ``parse_vla_failure_oscillation.py`` raises ``review_flag``.
The thresholds are imported from that module so the plot always matches the
documented values.

On the full 142-episode set the shaded box contains exactly the 7 oscillation
episodes and none of the 135 non-oscillation episodes (see the calibration note
in parse_vla_failure_oscillation.py).

Usage (run from ``lerobot_so101/data/``)::

    python plot_oscillation_separation.py
    python plot_oscillation_separation.py --worklist vla_failure_analysis/labeling_worklist.csv
    python plot_oscillation_separation.py -o vla_failure_analysis/plots
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
except ImportError:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)

from parse_vla_failure_oscillation import PEAK_THRESHOLD_DEG, VEL_THRESHOLD_DEG

DEFAULT_WORKLIST = Path("vla_failure_analysis/labeling_worklist.csv")
DEFAULT_OUT_DIR = Path("vla_failure_analysis/plots")

OSC_COLOR = "#c0392b"
NON_COLOR = "#2a78d6"
ZERO_COLOR = "#7f8c8d"
FLAG_COLOR = "#c0392b"

Point = tuple[float, float]


def load_labelled(path: Path) -> tuple[list[Point], list[Point], list[Point]]:
    """Return (oscillation, non_oscillation_moving, zero_movement) point lists.

    Points are (mean, peak) tuples. Oscillation membership comes from the
    ``manual_label`` ground truth; the non-oscillation episodes are then split
    into a zero-movement cluster (``oscillation_class == 'zero_movement'`` --
    the arm barely moves, so mean and peak are both near zero) and the rest.
    """
    osc: list[Point] = []
    non: list[Point] = []
    zero: list[Point] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            ep = (row.get("episode_id") or "").strip()
            label = (row.get("manual_label") or "").strip()
            if not ep or not label:
                continue
            try:
                mean = float(row["max_mean_speed"])
                peak = float(row["max_peak_speed"])
            except (KeyError, ValueError):
                continue
            if label == "oscillation":
                osc.append((mean, peak))
            elif (row.get("oscillation_class") or "").strip() == "zero_movement":
                zero.append((mean, peak))
            else:
                non.append((mean, peak))
    return osc, non, zero


def make_plot(osc: list[Point], non: list[Point], zero: list[Point],
              vel: float, peak: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    all_mean = [m for m, _ in osc + non + zero] or [0.0]
    all_peak = [p for _, p in osc + non + zero] or [0.0]
    x_max = max(all_mean) * 1.08
    y_max = max(all_peak) * 1.08

    # Shaded flag region: both gates passed.
    ax.add_patch(plt.Rectangle((vel, peak), x_max - vel, y_max - peak,
                               facecolor=FLAG_COLOR, alpha=0.10,
                               edgecolor=FLAG_COLOR, linewidth=0.5, zorder=0))

    # Gate lines.
    ax.axvline(vel, color=FLAG_COLOR, linestyle="--", linewidth=1.3, zorder=1)
    ax.axhline(peak, color=FLAG_COLOR, linestyle="--", linewidth=1.3, zorder=1)
    ax.text(vel, y_max, f" mean = {vel:g}", color=FLAG_COLOR, fontsize=9,
            va="top", ha="left")
    ax.text(x_max, peak, f"peak = {peak:g} ", color=FLAG_COLOR, fontsize=9,
            va="bottom", ha="right")

    if zero:
        zx, zy = zip(*zero)
        ax.scatter(zx, zy, marker="^", s=32, facecolor=ZERO_COLOR, alpha=0.6,
                   edgecolor=ZERO_COLOR, linewidth=0.5, zorder=2)
    if non:
        nx, ny = zip(*non)
        ax.scatter(nx, ny, marker="s", s=28, facecolor=NON_COLOR, alpha=0.55,
                   edgecolor=NON_COLOR, linewidth=0.5, zorder=2)
    if osc:
        ox, oy = zip(*osc)
        ax.scatter(ox, oy, marker="o", s=60, facecolor=OSC_COLOR, alpha=0.9,
                   edgecolor="#7a1c14", linewidth=1.0, zorder=3)

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("max mean per-step velocity (deg/step)")
    ax.set_ylabel("max peak per-step velocity (deg/step)")
    ax.set_title(
        f"Review-flag separation on {len(osc) + len(non) + len(zero)} labelled episodes")
    ax.grid(True, color="0.85", linewidth=0.6, zorder=0)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=OSC_COLOR,
               markeredgecolor="#7a1c14", markersize=9,
               label=f"oscillation ({len(osc)})"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=NON_COLOR,
               markeredgecolor=NON_COLOR, markersize=8,
               label=f"non-oscillation ({len(non)})"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=ZERO_COLOR,
               markeredgecolor=ZERO_COLOR, markersize=8,
               label=f"zero movement ({len(zero)})"),
        Patch(facecolor=FLAG_COLOR, alpha=0.10, edgecolor=FLAG_COLOR,
              label=f"flag region (mean >= {vel:g} & peak >= {peak:g})"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.9,
              fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the review-flag threshold separation on labelled episodes.")
    parser.add_argument("--worklist", default=str(DEFAULT_WORKLIST),
                        help=f"Labelled worklist CSV (default: {DEFAULT_WORKLIST})")
    parser.add_argument("-o", "--out-dir", default=str(DEFAULT_OUT_DIR),
                        help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--vel-threshold", type=float, default=VEL_THRESHOLD_DEG,
                        help=f"Mean-speed gate (default: {VEL_THRESHOLD_DEG})")
    parser.add_argument("--peak-threshold", type=float, default=PEAK_THRESHOLD_DEG,
                        help=f"Peak-speed gate (default: {PEAK_THRESHOLD_DEG})")
    args = parser.parse_args()

    worklist = Path(args.worklist)
    if not worklist.is_file():
        print(f"ERROR: worklist not found: {worklist}")
        sys.exit(1)

    osc, non, zero = load_labelled(worklist)
    if not osc and not non and not zero:
        print(f"ERROR: no labelled rows in {worklist}")
        sys.exit(1)

    out_path = Path(args.out_dir) / "threshold_separation.png"
    make_plot(osc, non, zero, args.vel_threshold, args.peak_threshold, out_path)

    # Report the split under the current gates. Zero-movement episodes are part
    # of the non-oscillation ground truth, so they count toward the FP tally.
    def flagged(pt: Point) -> bool:
        return pt[0] >= args.vel_threshold and pt[1] >= args.peak_threshold

    non_all = non + zero
    tp = sum(flagged(p) for p in osc)
    fp = sum(flagged(p) for p in non_all)
    print(f"Labelled: {len(osc)} oscillation, {len(non_all)} non-oscillation "
          f"(of which {len(zero)} zero-movement) -- {len(osc) + len(non_all)} total.")
    print(f"Gates: mean >= {args.vel_threshold:g}, peak >= {args.peak_threshold:g}")
    print(f"  flagged oscillation (TP)     = {tp}/{len(osc)}")
    print(f"  flagged non-oscillation (FP) = {fp}/{len(non_all)}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
