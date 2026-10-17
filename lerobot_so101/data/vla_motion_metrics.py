"""
Per-episode motion-quality metrics from an episode's actions.csv (commanded joints),
shared by the live preference eval and offline analysis. Pure Python (no numpy) so it
runs anywhere the CSV does.

Oscillation flag (calibrated on the 227 labelled vla_failure_test episodes, see
vla_failure_analysis2/threshold_separation.png), over the analysed joints
(shoulder_pan, shoulder_lift, elbow_flex, wrist_flex):

    flag_velocity = max mean |step| >= 2.3  AND  max peak |step| >= 100     (original)
    flag_rolling  = max 150-step rolling mean |step| >= 5.4  AND  peak >= 100
    flag          = flag_velocity OR flag_rolling

The whole-episode mean is diluted by calm stretches (an episode that oscillates for
20 s after a clean first pick stays under 2.3); the 150-step (~5 s) rolling mean is
not. On the labelled set the OR rule keeps the original result (7/7, 0 false
positives: non-oscillation rolling max 5.11, the lowest oscillation above it 5.71).
A 60-150 window sweep gave the widest margin at 150 steps; short windows (<= 30 steps)
separate worse because one large correction dominates them.

Phase split: the episode is also cut at the first PLACEMENT — the first gripper
release (opening >= RELEASE_RISE above its closed level) after a hold of >= HOLD_STEPS
closed steps, with shoulder_pan displaced toward the tray by >= TRAY_PAN_DELTA from its
start. The second phase (going for another object) is where degradation was seen, so
the rolling flag is reported per phase (the whole-episode mean branch is not: a short
phase's mean is not comparable with the calibration).
"""

import csv
from pathlib import Path

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]
MEAN_THRESHOLD = 2.3
PEAK_THRESHOLD = 100.0
ROLL_WINDOW = 150
ROLL_THRESHOLD = 5.4

# Placement detection (commanded gripper, % of range; shoulder_pan, degrees).
CLOSE_DROP = 10.0       # a close: gripper falls this far below its open peak
HOLD_STEPS = 15         # ... and stays closed this many steps (carrying)
RELEASE_RISE = 15.0     # a release: gripper rises this far above its closed level
TRAY_PAN_DELTA = 20.0   # ... with pan this far from its start, signed (tray at +pan)


def load_actions(path: Path | str) -> dict[str, list[float]]:
    """{joint: commanded positions, 'gripper': ..., 'elapsed': input_elapsed} from
    actions.csv."""
    cols = {j: [] for j in [*JOINTS, "gripper", "elapsed"]}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            for j in [*JOINTS, "gripper"]:
                v = row.get(f"act_{j}.pos", "")
                cols[j].append(float(v) if v not in ("", None) else float("nan"))
            e = row.get("input_elapsed", "")
            cols["elapsed"].append(float(e) if e not in ("", None) else float("nan"))
    return cols


def _rolling_max_mean(s: list[float], w: int) -> float:
    if not s:
        return 0.0
    if len(s) < w:
        return sum(s) / len(s)
    cur = best = sum(s[:w])
    for i in range(w, len(s)):
        cur += s[i] - s[i - w]
        best = max(best, cur)
    return best / w


def speed_metrics(cols: dict[str, list[float]], lo: int = 0, hi: int | None = None) -> dict:
    """Oscillation metrics over actions [lo, hi)."""
    mean = peak = roll = 0.0
    for j in JOINTS:
        p = [x for x in cols[j][lo:hi] if x == x]  # drop NaN
        s = [abs(p[i] - p[i - 1]) for i in range(1, len(p))]
        if not s:
            continue
        mean = max(mean, sum(s) / len(s))
        peak = max(peak, max(s))
        roll = max(roll, _rolling_max_mean(s, ROLL_WINDOW))
    fv = mean >= MEAN_THRESHOLD and peak >= PEAK_THRESHOLD
    fr = roll >= ROLL_THRESHOLD and peak >= PEAK_THRESHOLD
    n = len(cols[JOINTS[0]][lo:hi])
    return {"n_actions": n, "max_mean_speed": round(mean, 3), "max_peak_speed": round(peak, 2),
            f"max_roll{ROLL_WINDOW}_speed": round(roll, 3),
            "flag_velocity": fv, "flag_rolling": fr, "flag": fv or fr}


def first_placement(cols: dict[str, list[float]]) -> int | None:
    """Index of the first release over the tray (see module docstring), or None."""
    g, pan = cols["gripper"], cols["shoulder_pan"]
    if not g:
        return None
    pan0 = pan[0]
    open_peak, closed_min, hold = g[0], None, 0
    for i, x in enumerate(g):
        if closed_min is None:
            open_peak = max(open_peak, x)
            if x <= open_peak - CLOSE_DROP:
                closed_min, hold = x, 1
            continue
        if x >= closed_min + RELEASE_RISE:
            toward_tray = (pan[i] - pan0 >= TRAY_PAN_DELTA if TRAY_PAN_DELTA >= 0
                           else pan[i] - pan0 <= TRAY_PAN_DELTA)
            if hold >= HOLD_STEPS and toward_tray:
                return i
            open_peak, closed_min, hold = x, None, 0  # re-opened elsewhere: a regrasp
            continue
        closed_min, hold = min(closed_min, x), hold + 1
    return None


def episode_metrics(path: Path | str) -> dict:
    """Whole-episode metrics + the first-placement split and per-phase metrics."""
    cols = load_actions(path)
    split = first_placement(cols)
    out = {"whole": speed_metrics(cols), "split_index": split,
           "split_t": (round(cols["elapsed"][split], 2) if split is not None else None)}
    if split is None:
        out["first"], out["after"] = speed_metrics(cols), None
    else:
        out["first"], out["after"] = speed_metrics(cols, 0, split), speed_metrics(cols, split)
    return out


if __name__ == "__main__":
    import json
    import sys
    for p in sys.argv[1:]:
        print(p, json.dumps(episode_metrics(p), indent=1))
