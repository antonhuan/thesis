#!/usr/bin/env python3
"""
Oscillation Diagnostic Plots for VLA Action Logs
=================================================
Generates 3 diagnostic plots for a given episode's action CSV:
  1. Measured/Commanded joint positions over time (one row per joint)
  2. Commanded vs measured overlay (if both available)
  3. Velocity profile with major reversal markers

All joints present in the CSV are plotted (one subplot row each).

Usage:
    python plot_oscillation_diagnostics.py <actions.csv> [--output-dir DIR] [--title TITLE]
                                           [--joints j1,j2,...]

The script auto-detects two CSV formats:
  - pouch_study_analysis: columns include shoulder_pan, shoulder_lift, elbow_flex, ...
    (these are commanded positions; no separate measured columns)
  - vla_failure_test: columns include in_shoulder_pan.pos, act_shoulder_pan.pos, ...
    (in_ = measured input, act_ = commanded action)

Examples:
    python plot_oscillation_diagnostics.py actions/purse__067_purse.csv
    python plot_oscillation_diagnostics.py actions/purse__067_purse.csv --output-dir plots/ --title "purse_067"
    python plot_oscillation_diagnostics.py actions/purse__067_purse.csv --joints shoulder_lift,elbow_flex
"""

import argparse
import csv
import os
import sys
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


# Canonical joint order for the SO-101 arm. Used to order plots consistently;
# any additional joints found in the CSV are appended after these.
CANONICAL_JOINTS = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper',
]

# Metadata columns in the bare (pouch_study_analysis) format that are NOT joints.
BARE_META_COLS = {'action_n', 't', 'step', 'queue', 'dmax'}


def load_csv(path):
    """Load action CSV and return columns as numpy arrays + header list."""
    with open(path) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        data = {h: [] for h in headers}
        for row in reader:
            for h in headers:
                try:
                    data[h].append(float(row[h]) if row[h] != '' else np.nan)
                except (ValueError, TypeError):
                    data[h].append(np.nan)
    return {k: np.array(v) for k, v in data.items()}, headers


def detect_format(headers):
    """
    Detect CSV format.
    Returns: 'split' if in_/act_ columns exist (vla_failure_test format),
             'bare' if bare joint names (pouch_study_analysis format).
    """
    if any(h.startswith('in_') or h.startswith('act_') for h in headers):
        return 'split'
    elif any(j in headers for j in CANONICAL_JOINTS):
        return 'bare'
    else:
        print(f"ERROR: Unrecognized CSV format. Headers: {headers}")
        sys.exit(1)


def _order_joints(names):
    """Return joint names in canonical order, appending any unknown ones."""
    names = list(names)
    ordered = [j for j in CANONICAL_JOINTS if j in names]
    extra = [j for j in names if j not in CANONICAL_JOINTS]
    return ordered + extra


def detect_joints(headers, fmt):
    """
    Return the ordered list of joint names present in the CSV.
    - split: joints derived from act_*.pos / in_*.pos columns
    - bare:  header columns that aren't metadata
    """
    if fmt == 'split':
        names = set()
        for h in headers:
            for prefix in ('act_', 'in_'):
                if h.startswith(prefix):
                    name = h[len(prefix):]
                    if name.endswith('.pos'):
                        name = name[:-len('.pos')]
                    names.add(name)
    else:  # bare
        names = {h for h in headers if h not in BARE_META_COLS}
    return _order_joints(names)


def get_joint_data(data, headers, fmt, joints):
    """
    Extract per-joint time series.
    Returns dict:
      't'      -> time array
      'joints' -> list of joint names (in plot order)
      'cmd'    -> {joint: commanded array}
      'meas'   -> {joint: measured array or None}
    """
    result = {'joints': joints, 'cmd': {}, 'meas': {}}

    if 't' in data:
        result['t'] = data['t']
    else:
        result['t'] = np.arange(len(next(iter(data.values()))))

    for j in joints:
        if fmt == 'split':
            result['cmd'][j] = data.get(f'act_{j}.pos', data.get(f'act_{j}'))
            result['meas'][j] = data.get(f'in_{j}.pos', data.get(f'in_{j}'))
        else:  # bare
            result['cmd'][j] = data.get(j)
            result['meas'][j] = None

    return result


def _new_axes(n):
    """Create a figure with n stacked, x-shared subplots. Always returns a list."""
    height = max(3.0, 2.6 * n)
    fig, axes = plt.subplots(n, 1, figsize=(14, height), sharex=True)
    if n == 1:
        axes = [axes]
    return fig, list(axes)


def _joint_label(name):
    """Human-friendly joint label, e.g. 'shoulder_lift' -> 'Shoulder Lift'."""
    return name.replace('_', ' ').title()


def compute_velocity(positions, times):
    """Compute velocity (degrees/step) from position array."""
    vel = np.diff(positions)
    return vel


def find_major_reversals(positions, threshold_deg=15.0):
    """
    Find major direction reversals: direction changes that occur only after
    at least `threshold_deg` of travel since the last reversal.
    Returns indices of reversal points.
    """
    if len(positions) < 3:
        return []

    reversals = []
    last_rev_pos = positions[0]
    last_direction = None

    for i in range(1, len(positions)):
        delta = positions[i] - positions[i-1]
        if abs(delta) < 0.01:
            continue

        current_dir = 1 if delta > 0 else -1
        travel_since_rev = abs(positions[i] - last_rev_pos)

        if last_direction is not None and current_dir != last_direction:
            if travel_since_rev >= threshold_deg:
                reversals.append(i)
                last_rev_pos = positions[i]

        last_direction = current_dir

    return reversals


def plot_positions(jdata, title, output_path):
    """Plot 1: Joint positions over time (one row per joint)."""
    joints = jdata['joints']
    fig, axes = _new_axes(len(joints))
    t = jdata['t']

    for i, j in enumerate(joints):
        ax = axes[i]
        cmd = jdata['cmd'][j]
        meas = jdata['meas'][j]
        if meas is not None:
            ax.plot(t, meas, 'b-', linewidth=0.8, alpha=0.9, label='Measured')
            ax.plot(t, cmd, 'r-', linewidth=0.6, alpha=0.5, label='Commanded')
        else:
            ax.plot(t, cmd, 'b-', linewidth=0.8, label='Commanded')
        ax.set_ylabel(f'{_joint_label(j)} (°)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        rng = np.nanmax(cmd) - np.nanmin(cmd)
        ax.text(0.02, 0.95, f'Range: {rng:.1f}°', transform=ax.transAxes, va='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        if i == 0:
            ax.set_title(f'{title} — Joint Positions')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_commanded_vs_measured(jdata, title, output_path):
    """Plot 2: Commanded vs measured overlay (one row per joint)."""
    joints = jdata['joints']
    t = jdata['t']
    has_meas = any(jdata['meas'][j] is not None for j in joints)

    if not has_meas:
        # Bare format: plot commanded positions with velocity coloring.
        fig, axes = _new_axes(len(joints))
        for i, j in enumerate(joints):
            ax = axes[i]
            pos = jdata['cmd'][j]
            vel_abs = np.abs(np.diff(pos))
            hi = np.percentile(vel_abs, 75) if len(vel_abs) else 0.0
            for k in range(len(t) - 1):
                color = 'red' if vel_abs[k] > hi else 'blue'
                ax.plot(t[k:k+2], pos[k:k+2], color=color, linewidth=0.8, alpha=0.7)
            ax.set_ylabel(f'{_joint_label(j)} (°)')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(f'{title} — Commanded Positions (red = high velocity)')
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
        return

    fig, axes = _new_axes(len(joints))
    for i, j in enumerate(joints):
        ax = axes[i]
        cmd = jdata['cmd'][j]
        meas = jdata['meas'][j]
        if meas is None:
            # Joint has no measured stream; show commanded only.
            ax.plot(t, cmd, 'r--', linewidth=0.7, label='Commanded', alpha=0.8)
        else:
            ax.plot(t, meas, 'b-', linewidth=1.0, label='Measured', alpha=0.9)
            ax.plot(t, cmd, 'r--', linewidth=0.7, label='Commanded', alpha=0.6)
            error = cmd - meas
            ax2 = ax.twinx()
            ax2.fill_between(t, error, alpha=0.15, color='orange', label='Error')
            ax2.set_ylabel('Error (°)', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
        ax.set_ylabel(f'{_joint_label(j)} (°)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title(f'{title} — Commanded vs Measured')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_velocity_reversals(jdata, title, output_path, threshold_deg=15.0):
    """Plot 3: Velocity profile with major reversal markers (one row per joint)."""
    joints = jdata['joints']
    fig, axes = _new_axes(len(joints))
    t = jdata['t']

    # Cycle through a color list so each joint is visually distinct.
    palette = ['steelblue', 'darkorange', 'seagreen', 'crimson',
               'mediumpurple', 'saddlebrown', 'teal', 'darkgoldenrod']

    for i, j in enumerate(joints):
        ax = axes[i]
        # Per-action-step velocity: diff the COMMANDED stream, which updates every
        # action step. The measured stream (in_*) is sampled once per chunk and
        # repeated across that chunk's exec rows, so diffing it yields zeros within
        # a chunk and a single large spike at each chunk boundary. Fall back to
        # measured only when no commanded stream exists (e.g. bare format quirks).
        pos = jdata['cmd'][j] if jdata['cmd'][j] is not None else jdata['meas'][j]
        color = palette[i % len(palette)]
        vel = np.diff(pos)
        t_vel = t[:-1]

        ax.plot(t_vel, vel, color=color, linewidth=0.6, alpha=0.8)
        ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        ax.fill_between(t_vel, vel, alpha=0.2, color=color)

        reversals = find_major_reversals(pos, threshold_deg)
        for rev_idx in reversals:
            if rev_idx < len(t):
                ax.axvline(x=t[rev_idx], color='red', linewidth=1.0, alpha=0.6)

        n_rev = len(reversals)
        n_steps = len(pos) - 1
        rev_per_step = n_rev / n_steps if n_steps > 0 else 0
        rng = np.nanmax(pos) - np.nanmin(pos)

        ax.set_ylabel(f'{_joint_label(j)}\nVelocity (°/step)')
        ax.grid(True, alpha=0.3)
        stats_text = f'Major reversals: {n_rev}  |  rev/step: {rev_per_step:.3f}  |  Range: {rng:.1f}°'
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, va='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        if i == 0:
            ax.set_title(f'{title} — Velocity & Major Reversals (threshold={threshold_deg}°)')

    axes[-1].set_xlabel('Action step')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate oscillation diagnostic plots from a VLA action log CSV.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('csv_path', help='Path to the actions CSV file')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='Directory for output PNGs (default: same directory as input CSV)')
    parser.add_argument('--title', '-t', default=None,
                        help='Title prefix for plots (default: derived from filename)')
    parser.add_argument('--threshold', type=float, default=15.0,
                        help='Major reversal threshold in degrees (default: 15.0)')
    parser.add_argument('--joints', default=None,
                        help='Comma-separated joints to plot (default: all joints in CSV)')
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"ERROR: File not found: {args.csv_path}")
        sys.exit(1)

    # Derive title from filename if not provided
    if args.title is None:
        args.title = os.path.splitext(os.path.basename(args.csv_path))[0]

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.csv_path) or '.'
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading: {args.csv_path}")
    data, headers = load_csv(args.csv_path)
    fmt = detect_format(headers)
    print(f"  Format: {fmt} ({'in_/act_ columns' if fmt == 'split' else 'bare joint names'})")
    print(f"  Rows: {len(data[headers[0]])}")

    joints = detect_joints(headers, fmt)
    if args.joints:
        requested = [j.strip() for j in args.joints.split(',') if j.strip()]
        missing = [j for j in requested if j not in joints]
        if missing:
            print(f"  WARNING: requested joints not found in CSV: {missing}")
        joints = [j for j in requested if j in joints]
        if not joints:
            print("ERROR: none of the requested joints are present in the CSV.")
            sys.exit(1)
    print(f"  Joints ({len(joints)}): {', '.join(joints)}")

    jdata = get_joint_data(data, headers, fmt, joints)

    prefix = os.path.join(args.output_dir, args.title)

    print("\nGenerating plots...")
    plot_positions(jdata, args.title, f"{prefix}_positions.png")
    plot_commanded_vs_measured(jdata, args.title, f"{prefix}_cmd_vs_meas.png")
    plot_velocity_reversals(jdata, args.title, f"{prefix}_velocity_reversals.png",
                            threshold_deg=args.threshold)

    print(f"\nDone! 3 plots saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
