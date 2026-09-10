#!/usr/bin/env python3
"""
Oscillation Diagnostic Plots for VLA Action Logs
=================================================
Generates 3 diagnostic plots for a given episode's action CSV:
  1. Measured/Commanded joint positions over time (shoulder_lift & elbow_flex)
  2. Commanded vs measured overlay (if both available)
  3. Velocity profile with major reversal markers

Usage:
    python plot_oscillation_diagnostics.py <actions.csv> [--output-dir DIR] [--title TITLE]

The script auto-detects two CSV formats:
  - pouch_study_analysis: columns include shoulder_pan, shoulder_lift, elbow_flex, ...
    (these are commanded positions; no separate measured columns)
  - vla_failure_test: columns include in_shoulder_pan.pos, act_shoulder_pan.pos, ...
    (in_ = measured input, act_ = commanded action)

Examples:
    python plot_oscillation_diagnostics.py actions/purse__067_purse.csv
    python plot_oscillation_diagnostics.py actions/purse__067_purse.csv --output-dir plots/ --title "purse_067"
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
    if any(h.startswith('in_') for h in headers):
        return 'split'
    elif 'shoulder_lift' in headers:
        return 'bare'
    else:
        print(f"ERROR: Unrecognized CSV format. Headers: {headers}")
        sys.exit(1)


def get_joint_data(data, headers, fmt):
    """
    Extract shoulder_lift and elbow_flex time series.
    Returns dict with keys: t, sl_cmd, ef_cmd, sl_meas, ef_meas
    sl_meas/ef_meas are None if not available (bare format).
    """
    result = {}

    if 't' in data:
        result['t'] = data['t']
    else:
        result['t'] = np.arange(len(next(iter(data.values()))))

    if fmt == 'split':
        result['sl_cmd'] = data.get('act_shoulder_lift.pos', data.get('act_shoulder_lift'))
        result['ef_cmd'] = data.get('act_elbow_flex.pos', data.get('act_elbow_flex'))
        result['sl_meas'] = data.get('in_shoulder_lift.pos', data.get('in_shoulder_lift'))
        result['ef_meas'] = data.get('in_elbow_flex.pos', data.get('in_elbow_flex'))
    else:  # bare
        result['sl_cmd'] = data['shoulder_lift']
        result['ef_cmd'] = data['elbow_flex']
        result['sl_meas'] = None
        result['ef_meas'] = None

    return result


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
    """Plot 1: Joint positions over time."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    t = jdata['t']

    # Shoulder lift
    ax = axes[0]
    if jdata['sl_meas'] is not None:
        ax.plot(t, jdata['sl_meas'], 'b-', linewidth=0.8, alpha=0.9, label='Measured')
        ax.plot(t, jdata['sl_cmd'], 'r-', linewidth=0.6, alpha=0.5, label='Commanded')
    else:
        ax.plot(t, jdata['sl_cmd'], 'b-', linewidth=0.8, label='Commanded')
    ax.set_ylabel('Shoulder Lift (°)')
    ax.set_title(f'{title} — Joint Positions')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    rng = np.nanmax(jdata['sl_cmd']) - np.nanmin(jdata['sl_cmd'])
    ax.text(0.02, 0.95, f'Range: {rng:.1f}°', transform=ax.transAxes, va='top',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Elbow flex
    ax = axes[1]
    if jdata['ef_meas'] is not None:
        ax.plot(t, jdata['ef_meas'], 'b-', linewidth=0.8, alpha=0.9, label='Measured')
        ax.plot(t, jdata['ef_cmd'], 'r-', linewidth=0.6, alpha=0.5, label='Commanded')
    else:
        ax.plot(t, jdata['ef_cmd'], 'b-', linewidth=0.8, label='Commanded')
    ax.set_ylabel('Elbow Flex (°)')
    ax.set_xlabel('Time (s)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    rng = np.nanmax(jdata['ef_cmd']) - np.nanmin(jdata['ef_cmd'])
    ax.text(0.02, 0.95, f'Range: {rng:.1f}°', transform=ax.transAxes, va='top',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_commanded_vs_measured(jdata, title, output_path):
    """Plot 2: Commanded vs measured overlay (only for split format)."""
    if jdata['sl_meas'] is None:
        # For bare format, plot commanded positions with velocity coloring
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        t = jdata['t']

        for i, (joint_name, cmd_key) in enumerate([('Shoulder Lift', 'sl_cmd'), ('Elbow Flex', 'ef_cmd')]):
            ax = axes[i]
            pos = jdata[cmd_key]
            vel = np.diff(pos)
            vel_abs = np.abs(vel)

            # Color by velocity magnitude
            for j in range(len(t) - 1):
                color = 'red' if vel_abs[j] > np.percentile(vel_abs, 75) else 'blue'
                ax.plot(t[j:j+2], pos[j:j+2], color=color, linewidth=0.8, alpha=0.7)

            ax.set_ylabel(f'{joint_name} (°)')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(f'{title} — Commanded Positions (red = high velocity)')

        axes[1].set_xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {output_path}")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    t = jdata['t']

    for i, (joint_name, cmd_key, meas_key) in enumerate([
        ('Shoulder Lift', 'sl_cmd', 'sl_meas'),
        ('Elbow Flex', 'ef_cmd', 'ef_meas')
    ]):
        ax = axes[i]
        ax.plot(t, jdata[meas_key], 'b-', linewidth=1.0, label='Measured', alpha=0.9)
        ax.plot(t, jdata[cmd_key], 'r--', linewidth=0.7, label='Commanded', alpha=0.6)

        # Show error band
        error = jdata[cmd_key] - jdata[meas_key]
        ax2 = ax.twinx()
        ax2.fill_between(t, error, alpha=0.15, color='orange', label='Error')
        ax2.set_ylabel('Error (°)', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        ax.set_ylabel(f'{joint_name} (°)')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title(f'{title} — Commanded vs Measured')

    axes[1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_velocity_reversals(jdata, title, output_path, threshold_deg=15.0):
    """Plot 3: Velocity profile with major reversal markers."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Use measured if available, otherwise commanded
    sl_pos = jdata['sl_meas'] if jdata['sl_meas'] is not None else jdata['sl_cmd']
    ef_pos = jdata['ef_meas'] if jdata['ef_meas'] is not None else jdata['ef_cmd']
    t = jdata['t']
    duration = t[-1] - t[0] if len(t) > 1 else 1.0

    for i, (joint_name, pos, color) in enumerate([
        ('Shoulder Lift', sl_pos, 'steelblue'),
        ('Elbow Flex', ef_pos, 'darkorange')
    ]):
        ax = axes[i]
        vel = np.diff(pos)
        t_vel = t[:-1]

        # Plot velocity
        ax.plot(t_vel, vel, color=color, linewidth=0.6, alpha=0.8)
        ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        ax.fill_between(t_vel, vel, alpha=0.2, color=color)

        # Find and mark major reversals
        reversals = find_major_reversals(pos, threshold_deg)
        for rev_idx in reversals:
            if rev_idx < len(t_vel):
                ax.axvline(x=t[rev_idx], color='red', linewidth=1.0, alpha=0.6)

        n_rev = len(reversals)
        rev_per_s = n_rev / duration if duration > 0 else 0
        rng = np.nanmax(pos) - np.nanmin(pos)

        ax.set_ylabel(f'{joint_name}\nVelocity (°/step)')
        ax.grid(True, alpha=0.3)

        stats_text = f'Major reversals: {n_rev}  |  rev/s: {rev_per_s:.3f}  |  Range: {rng:.1f}°'
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, va='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        if i == 0:
            ax.set_title(f'{title} — Velocity & Major Reversals (threshold={threshold_deg}°)')

    axes[1].set_xlabel('Time (s)')
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

    jdata = get_joint_data(data, headers, fmt)

    prefix = os.path.join(args.output_dir, args.title)

    print("\nGenerating plots...")
    plot_positions(jdata, args.title, f"{prefix}_positions.png")
    plot_commanded_vs_measured(jdata, args.title, f"{prefix}_cmd_vs_meas.png")
    plot_velocity_reversals(jdata, args.title, f"{prefix}_velocity_reversals.png",
                            threshold_deg=args.threshold)

    print(f"\nDone! 3 plots saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
