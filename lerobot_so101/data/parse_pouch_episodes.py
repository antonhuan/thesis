#!/usr/bin/env python3
"""Parse the pouch-study episodes out of ``pouch_study/``.

Each ``pouch_study/<object>/<NNN_object>/`` directory is one single-object
pick-and-place episode.  Every directory contains:

  * ``actions.log``  -- same format as the ``runs/`` action logs
  * ``actions.csv``  -- structured per-timestep input/output joint data
  * camera PNGs      -- front and wrist frames per inference step

This script mirrors ``parse_episodes.py`` but adapted for the simpler
pouch-study layout (one episode per directory, no task.log / VLM judgement).

Outputs (written to ``pouch_study_analysis/``):

  * manifest.csv              one row per episode with metadata, a blank
                               ``manual_success`` column, and oscillation
                               analysis columns
  * actions/<episode_id>.csv  per-episode T x 6 action trajectory
  * plots/<episode_id>_positions.png           (if --plots)
  * plots/<episode_id>_cmd_vs_meas.png         (if --plots)
  * plots/<episode_id>_velocity_reversals.png  (if --plots)

Oscillation columns added to manifest.csv:

  * sl_major_rev, ef_major_rev, total_major_rev  -- major reversal counts
  * sl_range, ef_range      -- joint range of motion (degrees)
  * sl_rev_per_s, ef_rev_per_s, max_rev_per_s  -- reversals per second
  * sl_mean_v, ef_mean_v    -- mean absolute velocity (deg/s)
  * oscillation_class       -- 'oscillation', 'normal', or 'zero_movement'

Run from ``lerobot_so101/data/``:

    python parse_pouch_episodes.py              # analyse only
    python parse_pouch_episodes.py --plots      # + plots for oscillating episodes
    python parse_pouch_episodes.py --plot-all   # + plots for ALL episodes
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass, field
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

STUDY_DIR = Path("pouch_study")
OUT_DIR = Path("pouch_study_analysis")

# --- line patterns (identical to parse_episodes.py) ---------------------------
RE_EP_START = re.compile(
    r"=== episode start \| task: '(?P<task>.*)' \| duration: (?P<dur>[\d.]+)s ==="
)
RE_EP_END = re.compile(
    r"=== episode end \| (?P<dur>[\d.]+)s \| converged=(?P<conv>\w+) \| "
    r"actions=(?P<n>\d+)(?: \| reason=(?P<reason>\w+))? ==="
)
RE_ACTION = re.compile(
    r"\[action #(?P<n>\d+)\] t=(?P<t>[\d.]+)s step=(?P<step>\d+) "
    r"queue=(?P<queue>\d+) dmax=(?P<dmax>n/a|[\d.]+) \| (?P<joints>.+)$"
)


def parse_joints(segment: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for tok in segment.strip().split():
        name, _, val = tok.partition("=")
        name = name.replace(".pos", "")
        out[name] = float(val)
    return out


@dataclass
class Episode:
    index: int
    task: str
    duration_s: str = ""
    converged: str = ""
    termination_reason: str = ""
    num_actions: int = 0
    truncated: bool = False
    actions: list[dict] = field(default_factory=list)


def parse_actions_log(path: Path) -> list[Episode]:
    episodes: list[Episode] = []
    cur: Episode | None = None

    for line in path.read_text().splitlines():
        m = RE_EP_START.search(line)
        if m:
            cur = Episode(index=len(episodes), task=m.group("task"), duration_s=m.group("dur"))
            episodes.append(cur)
            continue

        m = RE_EP_END.search(line)
        if m and cur is not None:
            cur.duration_s = m.group("dur")
            cur.converged = m.group("conv")
            cur.termination_reason = m.group("reason") or ""
            cur.num_actions = int(m.group("n"))
            cur = None
            continue

        m = RE_ACTION.search(line)
        if m and cur is not None:
            joints = parse_joints(m.group("joints"))
            row = {
                "action_n": int(m.group("n")),
                "t": m.group("t"),
                "step": int(m.group("step")),
                "queue": int(m.group("queue")),
                "dmax": "" if m.group("dmax") == "n/a" else m.group("dmax"),
            }
            row.update({j: joints.get(j, "") for j in JOINTS})
            cur.actions.append(row)

    for ep in episodes:
        if ep.converged == "" and not ep.num_actions:
            ep.truncated = True
            ep.num_actions = len(ep.actions)
    return episodes


def count_chunks(csv_path: Path) -> int:
    """Count distinct chunk_ids in the actions.csv."""
    if not csv_path.exists():
        return 0
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        return len({row["chunk_id"] for row in reader})


def count_images(ep_dir: Path) -> int:
    return len(list(ep_dir.glob("*.png")))


# ---------------------------------------------------------------------------
# Oscillation analysis
# ---------------------------------------------------------------------------

REVERSAL_THRESHOLD_DEG = 15.0


def find_major_reversals(positions: np.ndarray, threshold_deg: float = REVERSAL_THRESHOLD_DEG) -> list[int]:
    """Direction changes only after >= threshold_deg travel since last reversal."""
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


def oscillation_metrics(actions: list[dict]) -> dict:
    """Compute oscillation metrics from an episode's parsed action rows."""
    if len(actions) < 2:
        return {
            "sl_major_rev": 0, "ef_major_rev": 0, "total_major_rev": 0,
            "sl_range": 0.0, "ef_range": 0.0,
            "sl_rev_per_s": 0.0, "ef_rev_per_s": 0.0, "max_rev_per_s": 0.0,
            "sl_mean_v": 0.0, "ef_mean_v": 0.0, "oscillation_class": "zero_movement",
        }

    t = np.array([float(a["t"]) for a in actions])
    sl = np.array([float(a["shoulder_lift"]) for a in actions])
    ef = np.array([float(a["elbow_flex"]) for a in actions])

    duration = t[-1] - t[0]

    sl_range = float(sl.max() - sl.min())
    ef_range = float(ef.max() - ef.min())

    sl_revs = find_major_reversals(sl)
    ef_revs = find_major_reversals(ef)

    dt = np.diff(t)
    dt[dt == 0] = np.nan
    sl_mean_v = float(np.nanmean(np.abs(np.diff(sl) / dt)))
    ef_mean_v = float(np.nanmean(np.abs(np.diff(ef) / dt)))

    sl_rev_per_s = len(sl_revs) / duration if duration > 0 else 0
    ef_rev_per_s = len(ef_revs) / duration if duration > 0 else 0
    max_rev_per_s = max(sl_rev_per_s, ef_rev_per_s)
    max_range = max(sl_range, ef_range)

    if max_range < 10:
        cls = "zero_movement"
    elif max_rev_per_s >= 0.5 and max_range > 50:
        cls = "oscillation"
    else:
        cls = "normal"

    return {
        "sl_major_rev": len(sl_revs),
        "ef_major_rev": len(ef_revs),
        "total_major_rev": len(sl_revs) + len(ef_revs),
        "sl_range": round(sl_range, 1),
        "ef_range": round(ef_range, 1),
        "sl_rev_per_s": round(sl_rev_per_s, 3),
        "ef_rev_per_s": round(ef_rev_per_s, 3),
        "max_rev_per_s": round(max_rev_per_s, 3),
        "sl_mean_v": round(sl_mean_v, 3),
        "ef_mean_v": round(ef_mean_v, 3),
        "oscillation_class": cls,
    }


OSC_FIELDS = [
    "sl_major_rev", "ef_major_rev", "total_major_rev",
    "sl_range", "ef_range",
    "sl_rev_per_s", "ef_rev_per_s", "max_rev_per_s",
    "sl_mean_v", "ef_mean_v",
    "oscillation_class",
]


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------

def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def generate_plots(actions: list[dict], title: str, plot_dir: Path) -> None:
    """Generate 3 oscillation diagnostic plots for one episode."""
    plt = _get_plt()
    plot_dir.mkdir(parents=True, exist_ok=True)

    t = np.array([float(a["t"]) for a in actions])
    sl = np.array([float(a["shoulder_lift"]) for a in actions])
    ef = np.array([float(a["elbow_flex"]) for a in actions])
    duration = t[-1] - t[0] if len(t) > 1 else 1.0

    # --- Plot 1: Joint positions ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for i, (name, pos) in enumerate([("Shoulder Lift", sl), ("Elbow Flex", ef)]):
        ax = axes[i]
        ax.plot(t, pos, "b-", lw=0.8, label="Commanded")
        ax.set_ylabel(f"{name} (°)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        rng = pos.max() - pos.min()
        ax.text(0.02, 0.95, f"Range: {rng:.1f}°", transform=ax.transAxes, va="top",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        if i == 0:
            ax.set_title(f"{title} — Joint Positions")
    axes[1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{title}_positions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 2: Velocity-coloured positions ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for i, (name, pos) in enumerate([("Shoulder Lift", sl), ("Elbow Flex", ef)]):
        ax = axes[i]
        vel_abs = np.abs(np.diff(pos))
        p75 = np.percentile(vel_abs, 75) if len(vel_abs) > 0 else 1
        for j in range(len(t) - 1):
            color = "red" if vel_abs[j] > p75 else "blue"
            ax.plot(t[j:j + 2], pos[j:j + 2], color=color, lw=0.8, alpha=0.7)
        ax.set_ylabel(f"{name} (°)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title(f"{title} — Commanded Positions (red = high velocity)")
    axes[1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{title}_cmd_vs_meas.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 3: Velocity + major reversals ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for i, (name, pos, color) in enumerate([
        ("Shoulder Lift", sl, "steelblue"), ("Elbow Flex", ef, "darkorange"),
    ]):
        ax = axes[i]
        vel = np.diff(pos)
        t_vel = t[:-1]
        ax.plot(t_vel, vel, color=color, lw=0.6, alpha=0.8)
        ax.axhline(y=0, color="black", lw=0.5, alpha=0.3)
        ax.fill_between(t_vel, vel, alpha=0.2, color=color)
        reversals = find_major_reversals(pos)
        for idx in reversals:
            if idx < len(t_vel):
                ax.axvline(x=t[idx], color="red", lw=1.0, alpha=0.6)
        n_rev = len(reversals)
        rev_per_s = n_rev / duration if duration > 0 else 0
        rng = pos.max() - pos.min()
        stats = f"Major reversals: {n_rev}  |  rev/s: {rev_per_s:.3f}  |  Range: {rng:.1f}°"
        ax.text(0.02, 0.95, stats, transform=ax.transAxes, va="top",
                fontsize=9, bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        ax.set_ylabel(f"{name}\nVelocity (°/step)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title(f"{title} — Velocity & Major Reversals (threshold={REVERSAL_THRESHOLD_DEG}°)")
    axes[1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(plot_dir / f"{title}_velocity_reversals.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Parse pouch-study episodes and analyse oscillation.")
    parser.add_argument("--plots", action="store_true",
                        help="Generate diagnostic plots for oscillating episodes")
    parser.add_argument("--plot-all", action="store_true",
                        help="Generate plots for ALL episodes (implies --plots)")
    args = parser.parse_args()
    if args.plot_all:
        args.plots = True

    actions_logs = sorted(STUDY_DIR.glob("*/*/actions.log"))

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "actions").mkdir(exist_ok=True)
    plot_dir = OUT_DIR / "plots"

    manifest_fields = [
        "episode_id", "object_type", "episode_dir", "task",
        "num_actions", "duration_s", "converged", "termination_reason",
        "truncated", "num_chunks", "num_images", "actions_csv",
        "manual_success", *OSC_FIELDS,
    ]

    manifest_rows: list[dict] = []
    truncated_ids: list[str] = []
    warnings: list[str] = []
    n_osc = 0
    n_plotted = 0

    for log_path in actions_logs:
        ep_dir = log_path.parent
        object_type = ep_dir.parent.name
        episode_dir_name = ep_dir.name
        episode_id = f"{object_type}__{episode_dir_name}"

        episodes = parse_actions_log(log_path)
        if not episodes:
            warnings.append(f"{object_type}/{episode_dir_name}: no episodes found in actions.log")
            continue
        if len(episodes) > 1:
            warnings.append(
                f"{object_type}/{episode_dir_name}: expected 1 episode, found {len(episodes)}"
            )

        ep = episodes[0]
        actions_rel = Path("actions") / f"{episode_id}.csv"

        with (OUT_DIR / actions_rel).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["action_n", "t", "step", "queue", "dmax", *JOINTS])
            w.writeheader()
            w.writerows(ep.actions)

        if ep.truncated:
            truncated_ids.append(episode_id)

        # Oscillation analysis
        osc = oscillation_metrics(ep.actions)

        if osc["oscillation_class"] == "oscillation":
            n_osc += 1

        # Diagnostic plots
        if args.plots and (args.plot_all or osc["oscillation_class"] == "oscillation"):
            generate_plots(ep.actions, episode_id, plot_dir)
            n_plotted += 1

        row = {
            "episode_id": episode_id,
            "object_type": object_type,
            "episode_dir": episode_dir_name,
            "task": ep.task,
            "num_actions": ep.num_actions,
            "duration_s": ep.duration_s,
            "converged": ep.converged,
            "termination_reason": ep.termination_reason,
            "truncated": ep.truncated,
            "num_chunks": count_chunks(ep_dir / "actions.csv"),
            "num_images": count_images(ep_dir),
            "actions_csv": str(actions_rel),
            "manual_success": "",
        }
        row.update(osc)
        manifest_rows.append(row)

    manifest_path = OUT_DIR / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=manifest_fields)
        w.writeheader()
        w.writerows(manifest_rows)

    # --- summary ---------------------------------------------------------------
    by_object = Counter(r["object_type"] for r in manifest_rows)
    converged_count = sum(1 for r in manifest_rows if r["converged"] == "True")
    print(f"Parsed {len(manifest_rows)} episodes from {len(actions_logs)} actions.log files.")
    print(f"By object: {', '.join(f'{k}={v}' for k, v in sorted(by_object.items()))}")
    print(f"Converged: {converged_count}/{len(manifest_rows)}")
    print(f"Oscillation: {n_osc}/{len(manifest_rows)} episodes classified as oscillating")
    print(f"Wrote {manifest_path} and {len(manifest_rows)} per-episode CSVs to {OUT_DIR / 'actions'}/.")
    if args.plots:
        print(f"Plots: {n_plotted} episodes × 3 plots in {plot_dir}/")
    if truncated_ids:
        print(
            f"{len(truncated_ids)} truncated episode(s) (no 'episode end' marker): "
            + ", ".join(truncated_ids)
        )
    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
