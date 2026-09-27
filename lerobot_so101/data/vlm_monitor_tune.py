"""
Interactive harness for tuning the two-pass semantic monitor.

The semantic monitor answers ONE question: is the arm going for the RIGHT object?
It runs in two passes, fed the INITIAL SCENE (the front camera's first frame, before
the arm moved in) plus a short sliding-window trajectory shown from BOTH cameras — the
front (overhead) and wrist (in-hand close-up) views for the same steps:

  Pass 1 (identify) — list the objects the arm's trajectory is CONVERGING toward /
      grasping (or 'unclear'). The window (not a single frame) is what lets the VLM tell
      an object the gripper is heading for from one it is merely passing over, which is
      needed to STOP a wrong reach BEFORE the grasp. The sub-task's target is
      deliberately withheld here: if you name it first ("is it grabbing the pouch?") the
      VLM hallucinates agreement, so we only ask the open question ("what is it going
      for?").
  Pass 2 (compare)  — given the sub-task and pass 1's plausible targets, CONTINUE as
      long as the target is one of them, else STOP. Only this pass is told the target,
      and it does pure identity comparison, no perception.

Two patches target the two false-positive modes the two-pass has (in /scan):
  UNCLEAR bias (reach ambiguity) — when two+ objects are plausible because they are
      close (one in front of/behind the gripper, or side by side) or overlapping, the
      identify pass answers 'unclear' rather than guessing, and compare fail-safes
      'unclear' -> CONTINUE. This stops the "wrong nearby object -> STOP" false
      positive at the source.
  GRASP COMMIT (placement overlap) — once a window confirms the RIGHT object was
      grasped (compare match=yes + held=yes, gated on the gripper having actually
      closed on it), the monitor COMMITS and retires the semantic check for the rest
      of the run. The object is in hand and the task is now placement, so a later
      "wrong object" read is only a tray-overlap distractor. It cannot mask a real
      wrong grasp: grabbing the wrong object gives match=no and STOPs before commit
      ever latches.

Motion quality (speed, oscillation, stalls, drift) is out of scope — a separate
kinematic monitor owns it. Drop detection was tried here and removed: from a single
frame a placement and a drop are indistinguishable, and it produced false STOPs.

This script is a focused REPL for iterating on that monitor: load the model once,
point it at a saved run of frames, type the sub-task, and inspect the verdict.
Tune the two prompts live (/edit) without re-running the robot, then paste the
winning prompts back into the orchestrator.

Input — a clip (a whole run of ordered full-res frames):
  --clip <glob|dir> : e.g. the per-timestep stills under
      vla_failure_test/<object>/<episode>/, holding both front_ and wrist_ frames.
      A single check feeds the front camera's first frame as the initial scene +
      the last --window frames from EACH view (front + wrist) as the trajectory.
      /scan (or --scan) slides that window across EVERY frame in the run — an
      offline replay of how the monitor fires across a live episode, ending in a
      verdict timeline. The sub-task can often be read off the filenames.

Shared model loading and inference live in vlm_core.py.

Requirements:
    pip install torch transformers accelerate pillow numpy
    pip install git+https://github.com/huggingface/transformers   # Qwen3-VL

Usage:
    # Tune against one saved run:
    python vlm_monitor_tune.py --clip vla_failure_test/banana/125_banana

    # Slide the window across the whole run on startup:
    python vlm_monitor_tune.py --clip vla_failure_test/banana/125_banana --scan

    # Load prompt overrides from a file you can edit + /reload:
    python vlm_monitor_tune.py --clip <dir> --prompt-file my_monitor_prompt.txt
"""

import argparse
import csv
import glob as globlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from vlm_core import load_model, generate

# Number of tokens each pass is allowed to generate. Matches the orchestrator's
# monitor call so verdict length here tracks production.
MONITOR_MAX_NEW_TOKENS = 384

# Motion gate. The identify pass cannot tell a staged/settled arm from one actively
# reaching when the arm is barely moving, so at the start (arm not yet moved in) and
# the end (arm settled/parked) it wrongly names the nearest object as the target. When
# the arm is not actively reaching we short-circuit to CONTINUE without asking the VLM.
#
# The JOINT gate (preferred; from the run's actions.csv) is a HYBRID of two measures,
# because the two ends look different in joint space:
#   START distance — L2 of the current pose from the EPISODE-START pose. Small while
#                    the arm is still near its home/staging pose -> catches startup.
#   STILL peak     — peak L2 displacement across the window relative to the window's
#                    first frame (a velocity-like measure). Small when the arm is
#                    barely moving right now -> catches the settled/parked end and
#                    brief grasp pauses, even though the arm is far from home there.
# The window is GATED if EITHER measure is below its threshold. This mirrors the live
# orchestrator's peak-L2 no_movement_threshold while adding home-proximity for startup.
#
# PIXEL gate (fallback, used only when a run has no joint CSV): mean inter-frame pixel
# difference (0-255). A single threshold, no home reference.
#
# 0 disables the relevant threshold. All are tunable live (/jstart, /jstill, /motion)
# and the measured values + which gate fired are printed on every check.
# Calibrated on 125_banana (parks at tray) + 158_plush (returns home).
# START is anchored to where the VLM's read stops being drawn to the object near the
# start pose: in 158_plush that is ~frame 8, dist ~73 from the episode-start pose. dist
# is keyed on the CURRENT (latest) frame, so the gate opens as soon as the newest frame
# passes ~frame 8 (first VLM window 5..8), not one window later.
# STILL must clear plush's settled-end peak (6.6) yet stay under banana's lowest
# reaching window (7.95) -> 7.0.
DEFAULT_JOINT_START_THRESH = 150   # current-frame L2 from episode start; below => near start
DEFAULT_JOINT_STILL_THRESH = 7.0    # within-window peak L2; below => not moving now
DEFAULT_MOTION_THRESH = 2.0         # pixel fallback

# Grasp-COMMIT gate. Once the two-pass has confirmed the arm grasped the RIGHT object,
# the monitor COMMITS to that identity and suppresses further STOPs for the rest of the
# run — the object is in hand and the task is now placement, so any later "wrong object"
# read is a tray-overlap distractor, not a real error (this is the placement-overlap
# false-positive fix). To make the commit fire on a REAL grasp (not a premature
# `held: yes` while the gripper is still wide open mid-approach), it is gated on the
# gripper having actually CLOSED on the object.
#
# The close test is SELF-CALIBRATING, so it is object-independent: the approach always
# opens the gripper to (near) its widest, then closes onto the object, so a grasp is
# when the current aperture sits at least this margin BELOW the widest aperture seen so
# far this run. A thin object closes a lot, a thick one less, but both close in from the
# run's own open peak — no fixed threshold that only fits one object. It also naturally
# requires the open->close cycle to have happened (so it cannot fire on the empty
# gripper at the very start). This gripper gate is the ONLY place joints are used. When
# a run has no joint CSV, the gate is skipped and the commit is VLM-only.
DEFAULT_GRASP_CLOSE_MARGIN = 6.0

# GRASP-IMMINENT gate (off by default; replaces the start/still/pixel gates when on).
# The live orchestrator pauses the arm while the VLM runs, so the VLA's queued action
# chunk is the policy's own plan for the next moments. Instead of guessing from motion
# when a reach is resolved, check ONLY when that queue says a grasp is about to happen:
# the gripper has opened (its aperture rose at least GRIPPER_OPEN_MIN above the
# episode-start aperture), it has not yet closed on anything, and the queued chunk
# plans to close it by at least this margin below the current aperture. That is the
# moment the wrist view shows one object under the open gripper. Once the first grasp
# is observed (aperture this margin below the run's open peak) the gate never reopens
# — the pre-grasp look is done. Offline, the queue is each frame's logged `act_*` rows
# (only the executed prefix of the chunk, so the horizon is shorter than live). 0
# disables. Tunable with --grasp-imminent-margin / /imminent.
DEFAULT_GRASP_IMMINENT_MARGIN = 0.0
GRIPPER_OPEN_MIN = 10.0

# APPROACH gate (off by default; replaces the start/still/pixel motion gate when on).
# Keys the reach check on the policy's own grasp cycle instead of a calibrated joint
# distance: the VLM is asked only from the frame where the queued (or observed) gripper
# aperture first rises >= this margin above the episode-start aperture (the approach
# opens the gripper — the queue plans it ~20 steps before it happens) until
# DEFAULT_APPROACH_GATE_TAIL frames after the first observed grasp close (the lift, where
# the reach check sees a wrong object in the gripper). The opening command is abrupt
# (aperture ~1 -> 15-30 in one chunk), so any margin in 3..15 opens on the same frame.
# 0 disables. Tunable with --approach-gate-margin / --approach-gate-tail.
DEFAULT_APPROACH_GATE_MARGIN = 0.0
DEFAULT_APPROACH_GATE_TAIL = 5
# CLOSE trigger for the approach gate (--approach-gate-trigger close): open instead at
# the first frame where the queue plans the grasp close within a FIXED look-ahead of
# DEFAULT_APPROACH_GATE_HORIZON control steps (~27.5 Hz, so 20 ~ 0.7 s): the queued
# aperture within the horizon drops >= DEFAULT_GRASP_CLOSE_MARGIN below the observed
# one, after the gripper has opened. A fixed horizon keeps the lead time independent of
# where the queue is in its refresh cycle, and stays within the logged (executed) part
# of each chunk offline. Falls back to the observed first close. Offline sweep: detection
# levels off at 15-20 steps; the opening trigger looked earlier and added false alarms.
DEFAULT_APPROACH_GATE_HORIZON = 20
# GRASP trigger (--approach-gate-trigger grasp): the monitor stays gated until the grasp
# and judges ONCE — the single reach window is the one ending at the first observed close
# (a look-back over the approach), followed by the usual hybrid verify. In the hybrid a
# pre-grasp STOP is deferred to the grasp anyway and a pre-grasp CONTINUE changes
# nothing, so the earlier checks were mostly redundant. Fallback when no close is
# observed: the window ending this many frames after the gripper opened (or the final
# window if it never opens), so a run without a grasp is still checked once.
DEFAULT_GRASP_FALLBACK_FRAMES = 6

# Pass-1 'reaching_for' values that mean the reach is not resolved.
_UNDECIDED_REACH = {"", "unclear", "none", "?", "unknown", "none (arm not moving)"}

# GRASP-VERIFY (hybrid; off by default). The reach check's STOP is treated as a
# SUSPICION, confirmed on the grasp itself: at the first observed close of the first
# grasp (opened >= GRIPPER_OPEN_MIN above start, then >= DEFAULT_GRASP_CLOSE_MARGIN
# below the open peak — the jaws are closed on the object, which is still on the
# table), a wrist-only forced-choice probe asks what is held, on that frame and the
# next. If both agree on an object and the compare pass says it IS the sub-task's
# target, the STOP is VETOED and the grasp is committed (check retired). A blind wrist
# ('nothing' / disagreement), no observed close, or a different object -> the STOP
# stands. Targets the reach pass's pre-grasp misreads (a flat object hidden under the
# arm, its neighbour in view), which the closed-jaw wrist view resolves.
# The veto compares PHYSICAL objects: the sub-task's object phrase is grounded once per
# run on the initial scene by match_task_object (closed choice over HELD_PROBE_OBJECTS),
# and the STOP is vetoed only when the wrist probe names that same object. Grounded to
# 'none' (the named object is not in the scene) -> never veto; 'ambiguous'/unparsed ->
# fall back to the text compare pass (grasp_verify_match="text" forces that fallback).
# The probe's object list is scene knowledge: fixed here for the tabletop set; live it
# must come from the scene (initial frame / task decomposition).
HELD_PROBE_OBJECTS = ["banana", "brown pouch", "plush moose toy"]
HELD_PROBE_PROMPT = (
    "This is a close-up from a camera mounted on a robot gripper; the two orange "
    "fingers are the gripper jaws. Which object is held between the jaws? Choose "
    "exactly one of: {choices}, nothing. "
    'Return ONLY JSON: {{"object": "<choice>", "reason": "<what you see>"}}')

# The wrist camera is mounted rotated 90° CLOCKWISE relative to the front (overhead)
# view, so a raw wrist frame's left/right/up/down do not match the table. We rotate
# every wrist frame by this many degrees COUNTER-CLOCKWISE at load time (PIL's positive
# rotation direction) so the two views share an orientation and the VLM can reason about
# which object is behind/beside which without mentally de-rotating. 0 disables (feed the
# wrist frames unrotated). Tunable with --wrist-rotate; flip the sign if the mounting is
# actually CCW.
DEFAULT_WRIST_ROTATE_CCW = 90

# Observed joint columns in actions.csv, in a fixed order (the 6 SO-101 joints).
_JOINT_COLS = ["in_shoulder_pan.pos", "in_shoulder_lift.pos", "in_elbow_flex.pos",
               "in_wrist_flex.pos", "in_wrist_roll.pos", "in_gripper.pos"]

# Index of the gripper aperture within a _JOINT_COLS vector (low=closed, high=open).
_GRIPPER_IDX = _JOINT_COLS.index("in_gripper.pos")


def current_gripper(joints: "list[list[float] | None] | None") -> float | None:
    """Gripper aperture of the CURRENT (latest) frame in a window, or None.

    Low is closed, high is open. Reads the last non-None joint vector, so it survives
    frames with no CSV row. Never raises.
    """
    try:
        if not joints:
            return None
        cur = next((j for j in reversed(joints) if j is not None), None)
        if cur is None or len(cur) <= _GRIPPER_IDX:
            return None
        return float(cur[_GRIPPER_IDX])
    except Exception:  # noqa: BLE001 - a gripper hint must never break a check
        return None


# ---------------------------------------------------------------------------
# Prompts (two-pass identify -> compare). Kept minimal on purpose.
#
#   identify -> Pass 1: name what the arm is going for, target-blind.
#   compare  -> Pass 2: is that the object the sub-task names?
# ---------------------------------------------------------------------------

IDENTIFY_SYSTEM_PROMPT = """You look at images from the cameras around a robot arm on a tabletop and answer questions about what the arm is doing, judging only from what you can actually see.

You are shown the tabletop's initial layout plus a short trajectory (several ordered frames, earliest first) from an overhead (front) camera and an in-hand (wrist) close-up. Watch how the gripper MOVES across the frames — where it is heading tells you what it is going for, which a single frame cannot. Reason like this:
1. Look at the objects on the table and work out which ones the arm could plausibly be reaching for — the ones the gripper's trajectory is actually travelling toward. Ignore objects the arm is clearly not going for; being on the table does not make something a candidate, and an object the gripper merely passes over on its way somewhere else is not the target.
2. For each plausible candidate, note the evidence: is the gripper physically converging on it or closing on it across the frames, or is it merely nearby / the largest / in the foreground? Proximity alone is NOT evidence of intent — an object can sit right next to the gripper while the arm is going for something else.
3. Commit to a single object ONLY when the gripper is clearly converging on or contacting that one object (the wrist close-up shows this best). If two or more candidates remain plausible — close together, overlapping, or one behind the other — the honest answer is that the target is unclear. Do not force a choice by defaulting to the nearest or biggest object.

The wrist close-up is oriented to match the front view, so directions agree between the two images. Judge which object is behind or beside which from the FRONT view, and use the wrist view for what is directly under / in the gripper.

Two ambiguous layouts deserve special caution — they are the classic way to name the wrong object:
  - BEHIND/OCCLUSION: the target sits directly behind another object along the gripper's line of approach. The nearer object looks like the one being reached for, but the gripper may be travelling to the one behind it. When one object is partly hidden by, or lines up behind, another under the descending gripper, treat BOTH as plausible.
  - SIDE BY SIDE: two objects sit right next to each other, almost touching, and the gripper descends into the gap or straddles both. Do not pick the one that happens to be a little closer to the frame's center or the gripper — if the gripper could close on either, treat both as plausible.

Treat two objects as plausible ONLY while the trajectory has not yet resolved which one the gripper is converging on. As soon as the motion across the frames clearly picks out one, drop the other and name it — an object the gripper only PASSES OVER on its way somewhere else is never a candidate, even if it sits right under or behind the gripper for a frame.

Being decisive about the wrong object is worse than admitting the target is ambiguous."""

COMPARE_SYSTEM_PROMPT = """You decide whether two short descriptions refer to the same physical object. Plain synonyms and descriptions count as the same thing (a "yellow fruit" is a banana)."""

DEFAULT_PROMPTS = {
    "identify": IDENTIFY_SYSTEM_PROMPT,
    "compare": COMPARE_SYSTEM_PROMPT,
}


def default_prompts() -> dict:
    return dict(DEFAULT_PROMPTS)


# ---------------------------------------------------------------------------
# JSON parsing (tolerant, mirrors the orchestrator's monitor() parse)
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        if t.endswith("```"):
            t = t[: -3]
    return t.strip()


def parse_json_object(output: str) -> dict | None:
    """Parse a JSON object from model output, tolerating code fences and prose.

    Returns the dict, or None if nothing parseable is found.
    """
    try:
        parsed = json.loads(_strip_code_fences(output))
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    text = output.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return None


# ---------------------------------------------------------------------------
# Motion gate
# ---------------------------------------------------------------------------

def _frame_index(name: str) -> int | None:
    """Leading numeric frame index of a filename, e.g. '00012_front_..png' -> 12."""
    m = re.match(r"(\d+)", Path(name).name)
    return int(m.group(1)) if m else None


def load_joint_states(run_dir: str) -> dict[int, list[float]]:
    """Map frame index -> observed 6-joint vector, read from <run_dir>/actions.csv.

    Uses the observed (`in_*.pos`) joints, which are constant across the exec rows of
    one input frame and so give one state per captured frame. Keyed by the numeric
    frame index (from the row's front_img) so it aligns to either camera view. Returns
    {} when there is no CSV or it lacks the joint columns — the caller then falls back
    to the pixel-diff gate. Never raises.
    """
    path = os.path.join(run_dir, "actions.csv")
    if not os.path.isfile(path):
        return {}
    try:
        states: dict[int, list[float]] = {}
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not all(c in reader.fieldnames for c in _JOINT_COLS):
                return {}
            for row in reader:
                idx = _frame_index(row.get("front_img", ""))
                if idx is None or idx in states:
                    continue  # first occurrence per frame is the observation at capture
                try:
                    states[idx] = [float(row[c]) for c in _JOINT_COLS]
                except (TypeError, ValueError):
                    continue
        return states
    except Exception:  # noqa: BLE001 - a motion hint must never break loading
        return {}


def load_planned_gripper(run_dir: str) -> dict[int, list[float]]:
    """Map frame index -> the gripper apertures the VLA queued at that frame.

    Reads the `act_gripper.pos` column of <run_dir>/actions.csv: the rows logged for
    one input frame are the actions executed from the chunk returned for it, in
    `exec_index` order, i.e. the queue's plan for the time until the next frame.
    Returns {} when there is no CSV or it lacks the columns. Never raises.
    """
    path = os.path.join(run_dir, "actions.csv")
    if not os.path.isfile(path):
        return {}
    try:
        plans: dict[int, list[tuple[int, float]]] = {}
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or not {"act_gripper.pos", "exec_index"} <= set(reader.fieldnames):
                return {}
            for row in reader:
                idx = _frame_index(row.get("front_img", ""))
                if idx is None:
                    continue
                try:
                    plans.setdefault(idx, []).append(
                        (int(row["exec_index"]), float(row["act_gripper.pos"])))
                except (TypeError, ValueError):
                    continue
        return {i: [g for _, g in sorted(v)] for i, v in plans.items()}
    except Exception:  # noqa: BLE001 - a gate hint must never break loading
        return {}


def joint_window_peak(joints: "list[list[float] | None]") -> float | None:
    """Peak L2 displacement across the window relative to its own first frame.

    A velocity-like "is the arm moving right now" measure: takes the window's first
    valid frame as the reference and returns the largest L2 distance any later frame
    reaches from it. Near 0 when the arm is settled/parked or paused (even far from
    home); large while reaching between waypoints. Returns None when fewer than two
    aligned frames are available.
    """
    try:
        valid = [np.asarray(j, dtype=float) for j in joints if j is not None]
        if len(valid) < 2:
            return None
        ref = valid[0]
        return max(float(np.linalg.norm(v - ref)) for v in valid[1:])
    except Exception:  # noqa: BLE001 - a motion hint must never break a check
        return None


def joint_dist_from_start(joints: "list[list[float] | None]",
                          episode_start: "list[float] | None") -> float | None:
    """L2 distance of the CURRENT (latest) pose from the EPISODE-START pose.

    Uses the window's last valid frame as "current" and `episode_start` (the run's
    very first observed joint vector, preserved across windowing) as the reference.
    Keying on the current frame — not the window min — means the gate opens the moment
    the NEWEST frame has moved far enough from the start pose, rather than waiting for
    the whole window to clear it (min lags by ~one window). With a threshold well above
    the early-approach distances it still gates startup (the current frame is small
    while the arm is near home) and the return-home tail (the current frame comes back
    near home). Returns None when the reference or all frames are unavailable.
    """
    try:
        if episode_start is None:
            return None
        current = next((j for j in reversed(joints) if j is not None), None)
        if current is None:
            return None
        return float(np.linalg.norm(np.asarray(current, dtype=float)
                                    - np.asarray(episode_start, dtype=float)))
    except Exception:  # noqa: BLE001 - a motion hint must never break a check
        return None


def clip_motion(clip: list[Image.Image]) -> float:
    """Mean absolute inter-frame pixel difference across the window (0-255 scale).

    Averages the frame-to-frame difference over every consecutive pair, so a window
    where the arm barely moved (staged at the start, settled/retreating at the end)
    scores near the camera-noise floor, while an arm crossing the frame scores well
    above it. Used to gate the semantic check: below a threshold the arm is not
    actively reaching, so naming a target is guesswork. Never raises.
    """
    try:
        if len(clip) < 2:
            return 0.0
        arrs = [np.asarray(im, dtype=np.float32) for im in clip]
        diffs = [float(np.abs(arrs[i + 1] - arrs[i]).mean())
                 for i in range(len(arrs) - 1)
                 if arrs[i + 1].shape == arrs[i].shape]
        return sum(diffs) / len(diffs) if diffs else 0.0
    except Exception:  # noqa: BLE001 - a motion hint must never break a check
        return 0.0


# ---------------------------------------------------------------------------
# User-message assembly
# ---------------------------------------------------------------------------

def _identify_content(clip: list[Image.Image],
                      initial: Image.Image | None,
                      fps: float,
                      wrist: "list[Image.Image | None] | None" = None,
                      held_hint: str | None = None) -> list[dict]:
    """Pass 1 content: INITIAL SCENE (front) + front & wrist trajectory.

    The initial scene is the front camera's opening frame. The sliding window is then
    fed as two aligned tracks — the FRONT (overhead) trajectory and the WRIST (in-hand
    close-up) trajectory — as ORDERED FULL-RESOLUTION STILLS (not a video clip): the
    Qwen video path shares a small pixel budget across all frames and downsamples each
    one hard, which blurred the object and drove mis-identification; individual images
    each get the full budget, so the object stays legible, and temporal order is carried
    in the per-frame labels. The window (not a single frame) is what lets the VLM see
    which object the gripper is CONVERGING on versus merely passing over — the signal
    needed to STOP a wrong reach before the grasp. `fps` is retained for signature
    compatibility and is unused. Deliberately omits the sub-task so the model names what
    it actually sees and is not led toward the target's name.
    """
    content: list[dict] = []
    if initial is not None:
        content += [
            {"type": "text", "text": "[INITIAL SCENE (front camera) — the tabletop "
             "at the start, before the arm moved in. Use it to see which objects are "
             "present and where each one lies.]"},
            {"type": "image", "image": initial},
        ]
    n = len(clip)
    content += [
        {"type": "text", "text": f"[FRONT TRAJECTORY — {n} overhead stills in time "
         "order, earliest first, latest last. Follow where the arm travels and what "
         "it converges on / lowers onto.]"},
    ]
    for i, im in enumerate(clip):
        tag = " (CURRENT — most recent)" if i == n - 1 else ""
        content += [
            {"type": "text", "text": f"front frame {i + 1}/{n}{tag}"},
            {"type": "image", "image": im},
        ]
    # Wrist track, aligned to the front window (and rotated at load to share the front
    # view's orientation). Feed only the frames that exist; skip the track entirely for
    # a run captured without a wrist camera.
    wrist_frames = [w for w in (wrist or []) if w is not None]
    if wrist_frames:
        m = len(wrist_frames)
        content += [
            {"type": "text", "text": f"[WRIST TRAJECTORY — the same {m} steps from the "
             "in-hand (wrist) camera, ordered stills, same time order, oriented to match "
             "the front view. This close-up shows the object directly under the gripper "
             "as it approaches and grasps.]"},
        ]
        for i, im in enumerate(wrist_frames):
            tag = " (CURRENT — most recent)" if i == m - 1 else ""
            content += [
                {"type": "text", "text": f"wrist frame {i + 1}/{m}{tag}"},
                {"type": "image", "image": im},
            ]
    if held_hint:
        content += [
            {"type": "text", "text": (
                f"[HELD-OBJECT MEMORY — in an earlier step the gripper closed on a "
                f"'{held_hint}'. It is most likely STILL that same object clamped in the "
                "gripper now. Use this only to avoid re-identifying the held object as "
                "something it is being lowered onto, set down near, or passes over — do "
                "not let overlapping tray objects change what you think is in the "
                "gripper. Still report what you actually see.]"
            )},
        ]
    content += [
        {"type": "text", "text": (
            "Which single object is the arm reaching for or grasping? Answer only "
            "from what you see across the frames. Judge by where the gripper is "
            "converging and what it actually closes on (the wrist close-up shows this "
            "best), not simply the largest or nearest object, and not an object the "
            "gripper only passes over on its way somewhere else.\n"
            "Answer 'unclear' whenever you cannot cleanly tell which object is the "
            "target — in particular when TWO OR MORE objects are plausible because they "
            "are close together (one in front of / behind the gripper, or two sitting "
            "side by side), or overlapping. In those cases prefer 'unclear' over "
            "guessing one of them; do NOT default to the foreground or nearest object. "
            "Only name a specific object when the trajectory clearly converges on it.\n"
            "Return ONLY this JSON:\n"
            "{\n"
            '  "candidates": ["<the objects the arm\'s trajectory is CONVERGING toward. '
            "If the frames clearly show the gripper converging on ONE object, list ONLY "
            "that object — even before it closes — and do NOT include an object it is "
            "merely passing over, or one that just sits nearby or behind without the "
            "gripper heading for it. List a SECOND object ONLY while it is genuinely "
            "still ambiguous which of the two the gripper is converging on (one directly "
            "behind the other along the approach, or two touching side by side, and the "
            "motion across the frames has NOT yet resolved which). The moment the "
            "trajectory resolves to one object, drop the other; once the gripper has "
            "closed (held=yes) this is always a SINGLE-object list. Whatever ends up in "
            "'reaching_for' must be one of these candidates, and a 'passed over' object "
            "must NOT be here.>\"],\n"
            '  "reaching_for": "<the one object the arm is going for — name it '
            "plainly (e.g. banana, pouch, plush toy), or 'unclear'>\",\n"
            '  "held": "<yes/no — is that object clamped in the closed gripper right now?>",\n'
            '  "reason": "<what in the frames tells you this>"\n'
            "}"
        )},
    ]
    return content


def _compare_content(sub_task: str, identified: dict,
                     target_object: str | None = None) -> list[dict]:
    """Pass 2 content: text identity check of the sub-task's target against the
    object the trajectory has resolved on. No image — identity only.

    The candidate set is collapsed by CONVERGENCE, not by grasp: once pass 1 has
    RESOLVED which object the gripper is converging on (`reaching_for` is a decided
    object, not 'unclear'), we compare the target against THAT one object alone — a
    'passed over' object that pass 1 may have left in `candidates` must not keep the
    target "in scope". So a resolved wrong reach STOPs immediately, BEFORE the grasp:
    reaching_for == target (or a synonym) -> CONTINUE, otherwise STOP. While pass 1 is
    still 'unclear' (genuinely ambiguous which object it is converging on) we keep the
    full candidate set and fail-safe to CONTINUE. This keys on convergence, not on
    `held`, which is what lets it STOP during the approach rather than at the grasp.
    """
    reaching = str(identified.get("reaching_for", "unclear")).strip()
    held = identified.get("held", "?")
    raw_cands = identified.get("candidates") or []
    if isinstance(raw_cands, str):
        raw_cands = [raw_cands]
    _undecided = {"", "unclear", "none", "?", "unknown", "none (arm not moving)"}
    committed = reaching.lower() not in _undecided

    cands = [str(c).strip() for c in raw_cands if str(c).strip()]
    if committed:
        # Trajectory resolved -> the decision is against the converged object alone,
        # regardless of any passed-over object still lingering in `candidates`.
        candidates = [reaching]
    else:
        # Still ambiguous -> keep the full plausible set (target stays in scope).
        candidates = cands
    cand_line = ", ".join(candidates) if candidates else "(none reported)"
    resolved_line = (
        f'The perception pass has RESOLVED the arm is going for: "{reaching}".\n'
        if committed else
        "The perception pass could not yet resolve which object the arm is going for "
        "(unclear).\n"
    )
    # GROUND-COMPARE: with `target_object` (the sub-task's object grounded on the initial
    # scene, e.g. "roll" -> "brown pouch"), pass 2 is told that object instead of the task
    # wording, so paraphrase resolution is done once by the grounder, not by pass 2.
    task_line = (f'The robot\'s target object is: "{target_object}".\n' if target_object
                 else f'The robot\'s sub-task is: "{sub_task}".\n')
    return [{"type": "text", "text": (
        task_line +
        f'A perception pass watched the arm\'s trajectory (held: {held}).\n'
        + resolved_line +
        f'The object(s) still in play are: {cand_line}.\n\n'
        "Decide whether the arm is still on track for the sub-task. A synonym or "
        "plain description of an object counts as the same object.\n"
        "- If the perception pass RESOLVED an object: return CONTINUE when the "
        "sub-task's target IS that object (or a synonym/description of it), and STOP "
        "when it is clearly a DIFFERENT object — the arm is converging on the wrong "
        "thing, stop it before it grasps.\n"
        "- If the perception pass is still 'unclear': the target is not ruled out, so "
        "return CONTINUE.\n"
        "Return ONLY this JSON:\n"
        "{\n"
        '  "target": "<the object the sub-task names>",\n'
        '  "acting_on": "<the object the arm is going for>",\n'
        '  "match": "<yes/no/unclear — does the object being reached for / held match the target>",\n'
        '  "target_is_candidate": "<yes/no — is the target the object the arm resolved on (or still-plausible while unclear)>",\n'
        '  "action": "CONTINUE" or "STOP",\n'
        '  "reason": "<why>"\n'
        "}"
    )}]


# ---------------------------------------------------------------------------
# Clip source: an ordered run of full-res frames + its initial scene
# ---------------------------------------------------------------------------

class ClipSource:
    """An ordered run of full-res frames. Holds the WHOLE run; a single check
    uses the last `window` frames as the trajectory, while /scan slides a window
    across all of them. The run's first frame is kept as the INITIAL SCENE and is
    fed on every look."""

    def __init__(self, clip: list[Image.Image], paths: list[str],
                 initial: Image.Image | None = None,
                 initial_path: str | None = None,
                 joints: "list[list[float] | None] | None" = None,
                 initial_joints: "list[float] | None" = None,
                 wrist: "list[Image.Image | None] | None" = None,
                 plans: "list[list[float] | None] | None" = None):
        self.clip = clip
        self.paths = paths
        # The WRIST-camera track, aligned index-for-index to `clip` (the front
        # track): wrist[i] is the wrist still captured at the same frame as
        # clip[i], or None when that frame has no wrist image. None for the whole
        # run when no wrist frames were found. Sliced alongside clip by
        # window()/tail() so a windowed check keeps both views in step.
        self.wrist = wrist
        # Per-frame observed joint vectors aligned to `clip`/`paths` (None entry for a
        # frame with no CSV row). None for the whole run when no actions.csv was found,
        # in which case the gate falls back to pixel motion.
        self.joints = joints
        # Per-frame QUEUED gripper plan aligned to `clip` (the act_gripper values the
        # VLA queued at that frame; None entry when missing). None for the whole run
        # without an actions.csv. Feeds the grasp-imminent gate.
        self.plans = plans
        # The EPISODE-START joint pose — the run's very first observed joint vector,
        # PRESERVED across window()/tail() so the startup (home-proximity) gate keeps a
        # fixed reference even on a windowed slice. Defaults to the first non-None
        # joints of this run.
        self.initial_joints = (initial_joints if initial_joints is not None
                               else next((j for j in (joints or []) if j is not None), None))
        # The INITIAL SCENE — the episode's very first frame, before the arm moved
        # in. Defaults to this run's first frame and is PRESERVED across
        # window()/tail() so a windowed check still sees the clean opening layout.
        self.initial = initial if initial is not None else (clip[0] if clip else None)
        self.initial_path = (initial_path if initial_path is not None
                             else (paths[0] if paths else None))

    def _slice_joints(self, sl: slice) -> "list[list[float] | None] | None":
        return self.joints[sl] if self.joints is not None else None

    def _slice_wrist(self, sl: slice) -> "list[Image.Image | None] | None":
        return self.wrist[sl] if self.wrist is not None else None

    def _slice_plans(self, sl: slice) -> "list[list[float] | None] | None":
        return self.plans[sl] if self.plans is not None else None

    def window(self, start: int, size: int) -> "ClipSource":
        """A sub-run of `size` frames starting at index `start`."""
        return ClipSource(self.clip[start:start + size],
                          self.paths[start:start + size],
                          initial=self.initial, initial_path=self.initial_path,
                          joints=self._slice_joints(slice(start, start + size)),
                          initial_joints=self.initial_joints,
                          wrist=self._slice_wrist(slice(start, start + size)),
                          plans=self._slice_plans(slice(start, start + size)))

    def tail(self, size: int) -> "ClipSource":
        """The last `size` frames — one monitor look at the end of the run."""
        return ClipSource(self.clip[-size:], self.paths[-size:],
                          initial=self.initial, initial_path=self.initial_path,
                          joints=self._slice_joints(slice(-size, None)),
                          initial_joints=self.initial_joints,
                          wrist=self._slice_wrist(slice(-size, None)),
                          plans=self._slice_plans(slice(-size, None)))


_TASK_RE = re.compile(r"(put_the_.+?)(?:\.(?:png|jpg|jpeg))?$", re.IGNORECASE)


def task_from_path(path: str) -> str:
    """Recover a sub-task from a frame filename like
    '00012_front_put_the_toy_on_the_tray.png' -> 'put the toy on the tray'."""
    m = _TASK_RE.search(Path(path).stem + Path(path).suffix)
    return m.group(1).replace("_", " ") if m else ""


def load_clip(pattern: str, view: str = "front", wrist_view: str = "wrist",
              wrist_rotate: float = DEFAULT_WRIST_ROTATE_CCW) -> ClipSource:
    """Load the WHOLE ordered run of frames matching a glob or directory.

    The FRONT view (`view`) is the primary track: it fixes the ordering and the
    frame indices, supplies the initial scene, and is the front half of the
    sliding window. The WRIST view (`wrist_view`) is loaded in parallel and
    aligned to the front frames by numeric frame index (a None entry for any front
    frame with no matching wrist still), so a check can feed both close-up and
    overhead views. Each wrist frame is rotated `wrist_rotate` degrees
    counter-clockwise at load (default DEFAULT_WRIST_ROTATE_CCW) so it shares the
    front view's orientation — the wrist camera is mounted 90° clockwise, so the
    raw frame's directions otherwise mislead the layout reasoning. 0 leaves them
    unrotated. When the names carry no view marker, everything is treated as the
    front track and there is no wrist track.

    Sorts by the numeric frame index in the name and keeps ALL matching frames —
    the consumer decides the window (a single check uses the tail; /scan slides
    across all of them). Also loads per-frame joint states from actions.csv in the
    run dir (when present) so the motion gate can use joint diffs. Raises
    ValueError with a clear message if nothing matches.
    """
    orig = pattern
    if os.path.isdir(pattern):
        pattern = os.path.join(pattern, "*")
    all_paths = [p for p in globlib.glob(pattern)
                 if p.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not all_paths:
        raise ValueError(f"No image frames matched: {pattern}")

    # Split by camera view. If no name carries the front marker, treat every frame
    # as the front track (a run without per-view names) and skip the wrist track.
    front_paths = [p for p in all_paths if view and view in Path(p).name]
    wrist_paths = [p for p in all_paths if wrist_view and wrist_view in Path(p).name]
    if not front_paths:
        front_paths = all_paths
        wrist_paths = []

    front_paths = sorted(
        front_paths,
        key=lambda p: (_frame_index(p) if _frame_index(p) is not None else 0))
    clip = [Image.open(p).convert("RGB") for p in front_paths]

    # Align wrist frames to the front frames by numeric index. A front frame with
    # no matching wrist still gets a None slot so the two tracks stay index-aligned.
    wrist = None
    if wrist_paths:
        wrist_by_idx = {}
        for p in wrist_paths:
            idx = _frame_index(p)
            if idx is not None and idx not in wrist_by_idx:
                wrist_by_idx[idx] = p
        def _load_wrist(path: str) -> Image.Image:
            im = Image.open(path).convert("RGB")
            # Rotate CCW to undo the wrist camera's 90° CW mounting so it lines up
            # with the front view. expand=True keeps the whole frame (no crop).
            return im.rotate(wrist_rotate, expand=True) if wrist_rotate % 360 else im

        wrist = [
            (_load_wrist(wrist_by_idx[_frame_index(p)])
             if _frame_index(p) in wrist_by_idx else None)
            for p in front_paths
        ]
        aligned = sum(1 for w in wrist if w is not None)
        print(f"  (loaded wrist view for {aligned}/{len(front_paths)} front frames"
              + (f", rotated {wrist_rotate:g}° CCW to match front)" if wrist_rotate % 360
                 else ")"))

    # Joint states live in the run directory alongside the frames.
    run_dir = orig if os.path.isdir(orig) else os.path.dirname(front_paths[0])
    states = load_joint_states(run_dir)
    joints = ([states.get(_frame_index(p)) for p in front_paths] if states else None)
    if states:
        aligned = sum(1 for j in joints if j is not None)
        print(f"  (loaded joint states for {aligned}/{len(front_paths)} frames from "
              f"{os.path.join(run_dir, 'actions.csv')})")

    planned = load_planned_gripper(run_dir)
    plans = ([planned.get(_frame_index(p)) for p in front_paths] if planned else None)

    return ClipSource(clip, [str(p) for p in front_paths], joints=joints, wrist=wrist,
                      plans=plans)


# ---------------------------------------------------------------------------
# The two-pass check
# ---------------------------------------------------------------------------

def _fmt(v: float | None) -> str:
    return f"{v:.1f}" if v is not None else "—"


def evaluate_gate(source: ClipSource, start_thresh: float, still_thresh: float,
                  motion_thresh: float) -> dict:
    """Decide whether the window should be motion-gated (skip the VLM, CONTINUE).

    Prefers the JOINT hybrid when joint states are available: gate if the arm is near
    the EPISODE-START pose (START distance below `start_thresh` -> startup) OR barely
    moving within the window (STILL peak below `still_thresh` -> settled/paused end).
    Falls back to a single PIXEL threshold when there is no joint CSV. A threshold of
    0 disables that individual condition. Returns a dict with the chosen signal, the
    measured metrics, whether it gated, and a human-readable summary + reason.
    """
    if source.joints is not None and (start_thresh > 0 or still_thresh > 0):
        dist = joint_dist_from_start(source.joints, source.initial_joints)
        peak = joint_window_peak(source.joints)
        if dist is not None or peak is not None:
            conds = []
            if dist is not None and start_thresh > 0 and dist < start_thresh:
                conds.append(f"near home (dist {dist:.1f} < {start_thresh:g})")
            if peak is not None and still_thresh > 0 and peak < still_thresh:
                conds.append(f"barely moving (peak {peak:.1f} < {still_thresh:g})")
            gated = bool(conds)
            return {
                "signal": "joint", "gated": gated,
                "dist": dist, "peak": peak, "motion": peak,
                "summary": f"joint dist={_fmt(dist)} peak={_fmt(peak)} "
                           f"(gates {start_thresh:g}/{still_thresh:g})",
                "reason": ("; ".join(conds) if gated
                           else "arm has left home and is actively moving"),
            }
    # PIXEL fallback
    pm = clip_motion(source.clip)
    gated = motion_thresh > 0 and pm < motion_thresh
    return {
        "signal": "pixel", "gated": gated,
        "dist": None, "peak": None, "motion": pm,
        "summary": (f"pixel motion={pm:.2f} (gate {motion_thresh:g})"
                    if motion_thresh > 0 else f"pixel motion={pm:.2f} (gate off)"),
        "reason": (f"pixel motion {pm:.2f} below gate {motion_thresh:g}" if gated
                   else "pixel motion above gate"),
    }


def run_check(model, processor, prompts, source: ClipSource, sub_task,
              temperature, fps=2.0, motion_thresh=DEFAULT_MOTION_THRESH,
              start_thresh=DEFAULT_JOINT_START_THRESH,
              still_thresh=DEFAULT_JOINT_STILL_THRESH,
              held_hint: str | None = None,
              gripper_closed: bool | None = None,
              target_object: str | None = None):
    """Two-pass identify->compare check over a clip, with pretty-printing.

    A motion gate runs first: if the arm is not actively reaching, we short-circuit to
    CONTINUE without asking the VLM — this is what stops the monitor from naming the
    object in front of a motionless arm as its target at the start and end of a run.
    The gate prefers a JOINT hybrid (near the episode-start pose => startup, or barely
    moving within the window => settled end; from actions.csv) and falls back to PIXEL
    motion when a run has no joint CSV. Then pass 1 identifies the object the arm is
    going for (target-blind) and pass 2 compares it to the sub-task. Returns the merged
    verdict dict (or {} on parse failure) so /scan can log it.

    `held` is KINEMATIC, not the VLM's guess: whether the fingers have actually closed
    on an object is a joint fact, not something to read off a frame (the VLM reports
    'held: yes' prematurely while the gripper is still at its approach-open). When
    `gripper_closed` is provided (the scan's self-calibrating aperture test — the
    aperture has come in from the run's open peak), it OVERRIDES the identify pass's
    `held`, and the VLM's own read is preserved as `held_vlm` for inspection. When it is
    None (no joint CSV, or a standalone single check with no run-level open peak) the
    VLM's `held` is used as-is. The kinematic `held` feeds the grasp-COMMIT (retire the
    check after a confirmed correct grasp); it does NOT gate the STOP — pass 2 can STOP a
    wrong reach during the approach, which is the point of watching the trajectory.
    """
    clip = source.clip
    initial = source.initial
    wrist = source.wrist
    n_wrist = sum(1 for w in (wrist or []) if w is not None)
    g = evaluate_gate(source, start_thresh, still_thresh, motion_thresh)
    mode = (f"clip x{len(clip)}"
            + (f" + wrist x{n_wrist}" if n_wrist else "")
            + (" + initial scene" if initial is not None else "")
            + f"  |  {g['summary']}")

    print(f"\n{'='*60}")
    print(f"SEMANTIC MONITOR (two-pass identify->compare) — {mode}")
    print(f"Sub-task: \"{sub_task}\"")
    print(f"{'='*60}")

    # --- Motion gate: too little movement to be reaching for anything ---
    if g["gated"]:
        print(f"  [motion gate] {g['reason']} — skipping VLM, CONTINUE.")
        gated = {
            "reaching_for": "none (arm not moving)",
            "held": "?",
            "target": sub_task,
            "acting_on": "none (arm not moving)",
            "match": "unclear",
            "action": "CONTINUE",
            "reason": f"motion-gated: {g['reason']}.",
            "motion": g["motion"],
            "motion_signal": g["signal"],
            "dist_from_start": g["dist"],
            "window_peak": g["peak"],
        }
        print(f"\n{'-'*60}\n  VERDICT: CONTINUE (motion-gated)\n{'-'*60}")
        return gated

    # --- Pass 1: identify (target-blind) ---
    id_messages = [
        {"role": "system", "content": [{"type": "text", "text": prompts["identify"]}]},
        {"role": "user", "content": _identify_content(clip, initial, fps, wrist=wrist,
                                                       held_hint=held_hint)},
    ]
    print("  [pass 1 — identify object (target-blind)]")
    id_output = generate(model, processor, id_messages,
                         max_new_tokens=MONITOR_MAX_NEW_TOKENS, temperature=temperature)
    identified = parse_json_object(id_output) or {}

    # --- Kinematic held: the joints, not the VLM, decide whether the gripper has
    # actually closed on an object. Override the identify pass's guess (kept as
    # held_vlm) so pass 2's grasp-gated STOP fires only on a real closed grasp. ---
    if gripper_closed is not None:
        identified["held_vlm"] = identified.get("held", "?")
        identified["held"] = "yes" if gripper_closed else "no"
        print(f"  [kinematic held] gripper {'CLOSED' if gripper_closed else 'open'} "
              f"-> held={identified['held']} (VLM said {identified['held_vlm']})")

    # --- Pass 2: compare to the sub-task ---
    cmp_messages = [
        {"role": "system", "content": [{"type": "text", "text": prompts["compare"]}]},
        {"role": "user", "content": _compare_content(sub_task, identified, target_object)},
    ]
    print("  [pass 2 — compare to sub-task]")
    cmp_output = generate(model, processor, cmp_messages,
                          max_new_tokens=MONITOR_MAX_NEW_TOKENS, temperature=temperature)
    verdict = parse_json_object(cmp_output) or {}

    merged = dict(identified)
    merged.update(verdict)
    if merged:
        merged["motion"] = g["motion"]
        merged["motion_signal"] = g["signal"]
        merged["dist_from_start"] = g["dist"]
        merged["window_peak"] = g["peak"]

    print(f"\nModel output:\n--- pass 1 (identify) ---\n{id_output}\n\n"
          f"--- pass 2 (compare) ---\n{cmp_output}")

    if not merged:
        print("\n[WARNING] Could not parse a JSON verdict — production would "
              "fail-safe to CONTINUE here.")
        return {}

    action = str(merged.get("action", "?")).upper()
    print(f"\n{'-'*60}")
    print(f"  VERDICT: {action}")
    print(f"  reaching_for: {merged.get('reaching_for', merged.get('acting_on', '?'))}")
    _cands = merged.get('candidates') or []
    if isinstance(_cands, str):
        _cands = [_cands]
    print(f"  candidates:   {', '.join(str(c) for c in _cands) if _cands else '?'}")
    _held_vlm = merged.get('held_vlm')
    print(f"  held:         {merged.get('held', '?')}"
          + (f"  (kinematic; VLM said {_held_vlm})" if _held_vlm is not None else ""))
    print(f"  target:       {merged.get('target', '?')}")
    print(f"  match:        {merged.get('match', '?')}")
    print(f"  target_cand:  {merged.get('target_is_candidate', '?')}")
    print(f"  reason:       {merged.get('reason', '?')}")
    print(f"{'-'*60}")
    return merged


def grasp_imminent_windows(source: ClipSource, starts: list[int], win: int,
                           margin: float) -> "list[tuple[bool, str]] | None":
    """Grasp-imminent gate decision for each window start (see
    DEFAULT_GRASP_IMMINENT_MARGIN): (open, reason) per window, keyed on the window's
    LATEST frame. Open only while the gripper has opened, has not yet closed on its
    first grasp, and the queued chunk plans to close it by >= `margin`. Depends only on
    joints + queued plans (never the VLM), so gate-only sweeps can call it. Returns
    None when the run lacks joint or plan data.
    """
    if source.joints is None or source.plans is None or source.initial_joints is None:
        return None
    start_ap = source.initial_joints[_GRIPPER_IDX]
    peak = None
    grasped = False
    out = []
    for start in starts:
        last = start + win - 1
        cur = current_gripper(source.joints[start:last + 1])
        plan = source.plans[last]
        if cur is not None:
            peak = cur if peak is None else max(peak, cur)
        opened = cur is not None and peak - start_ap >= GRIPPER_OPEN_MIN
        if opened and cur <= peak - margin:
            grasped = True  # the first grasp has closed -> the pre-grasp look is over
        drop = (cur - min(plan)) if plan and cur is not None else None
        if grasped:
            out.append((False, "first grasp already closed"))
        elif not opened:
            out.append((False, "gripper not opened yet"))
        elif drop is None or drop < margin:
            out.append((False, f"no close queued (planned drop {_fmt(drop)} < {margin:g})"))
        else:
            out.append((True, f"grasp queued: aperture {_fmt(cur)} -> planned min "
                              f"{_fmt(min(plan))} (drop {_fmt(drop)})"))
    return out


def first_grasp_close(source: ClipSource,
                      margin: float = DEFAULT_GRASP_CLOSE_MARGIN) -> int | None:
    """Index of the first frame where the gripper has closed on its first grasp:
    opened >= GRIPPER_OPEN_MIN above the episode-start aperture, then come in >=
    `margin` below the open peak. None without joints or if it never closes."""
    if source.joints is None or source.initial_joints is None:
        return None
    start = source.initial_joints[_GRIPPER_IDX]
    peak = None
    for i, j in enumerate(source.joints):
        if j is None:
            continue
        g = j[_GRIPPER_IDX]
        peak = g if peak is None else max(peak, g)
        if peak - start >= GRIPPER_OPEN_MIN and g <= peak - margin:
            return i
    return None


# Task -> scene-object matcher (tested by vlm_object_matcher_test.py): grounds the
# sub-task's object phrase on the INITIAL SCENE so the grasp-verify veto compares
# physical objects, not words ("the roll" -> the brown pouch).

def object_phrase(task: str) -> str:
    """'put the brown pouch on the tray' -> 'brown pouch' (tolerates the truncated and
    misspelled destinations found in the filenames)."""
    t = task.lower().strip()
    t = re.sub(r"^put the ", "", t)
    t = re.sub(r" (on|in) (the|thet)( \w*)?$", "", t)
    t = re.sub(r" (on|in)$", "", t)
    return t.strip()


LIST_PROMPT = (
    "This is a robot's tabletop workspace before the robot starts. List every distinct "
    "object on the table that the robot arm could pick up — not the robot itself, not "
    "the table, and not a container that objects are placed into. Use a short, plain "
    'name for each. Return ONLY JSON: {"objects": ["<name>", ...]}')

# Scene vocabulary for grasp_verify_match="inventory": built ONCE from the initial scene
# and used as the canonical terms for both the task grounding and the wrist probe, so
# nothing is hard-coded and the wrist never names objects from scratch. The names only
# need to be consistent between the two uses, not correct ("sausage" for the pouch works
# as long as grounding and wrist both use it).
INVENTORY_PROMPT = (
    "This is a robot's tabletop workspace before the robot starts. List every distinct "
    "object on the table that the robot arm could pick up — not the robot itself, not "
    "the table, and not a container that objects are placed into. For each give a short, "
    "plain, distinct name and a few words on what it looks like (colour, shape, material). "
    'Return ONLY JSON: {"objects": [{"name": "<name>", "appearance": "<looks>"}, ...]}')


def scene_inventory(model, processor, initial: Image.Image,
                    temperature: float = 0.1) -> list[dict]:
    """The pickable objects in the initial scene as [{"name", "appearance"}], names
    lower-cased and de-duplicated (order kept)."""
    msgs = [{"role": "user", "content": [{"type": "image", "image": initial},
                                         {"type": "text", "text": INVENTORY_PROMPT}]}]
    out = parse_json_object(generate(model, processor, msgs, max_new_tokens=256,
                                     temperature=temperature)) or {}
    inv, seen = [], set()
    for o in out.get("objects") or []:
        if isinstance(o, str):
            o = {"name": o, "appearance": ""}
        name = str(o.get("name", "")).strip().lower()
        if name and name not in seen and name not in ("none", "nothing", "ambiguous"):
            seen.add(name)
            inv.append({"name": name, "appearance": str(o.get("appearance", "")).strip()})
    return inv


def match_name(answer: str, names: list[str]) -> str | None:
    """Map a free answer back to one of `names` (exact, then containment either way)."""
    a = (answer or "").strip().lower()
    if a in names:
        return a
    hits = [n for n in names if n in a or a in n] if a else []
    return max(hits, key=len) if hits else None


MATCH_PROMPT = (
    "This is a robot's tabletop workspace before the robot starts. The robot's "
    'instruction is: "{task}".\n'
    "Which object on the table does the instruction's object ('{phrase}') refer to? "
    "Judge by what each object looks like; the instruction may use an unusual or loose "
    "description. The destination named in the instruction is not the object to pick.\n"
    "Choose exactly one of: {choices}, none (nothing on the table fits the "
    "description), ambiguous (more than one object fits equally well).\n"
    'Return ONLY JSON: {{"object": "<choice>", "reason": "<why>"}}')


def match_task_object(model, processor, initial: Image.Image, task: str,
                      objects: list[str] | None = None,
                      temperature: float = 0.1) -> dict:
    """Resolve the task's object phrase to one object in the initial scene.

    With `objects` (closed mode) the VLM chooses among them; without (open mode) it
    first lists the pickable objects it sees and chooses among its own list. Returns
    {"object": <choice | 'none' | 'ambiguous'>, "reason": ..., "scene_objects": [...]}.
    """
    scene = objects
    if scene is None:
        msgs = [{"role": "user", "content": [{"type": "image", "image": initial},
                                             {"type": "text", "text": LIST_PROMPT}]}]
        listed = parse_json_object(generate(model, processor, msgs, max_new_tokens=128,
                                            temperature=temperature)) or {}
        scene = [str(o).strip() for o in (listed.get("objects") or []) if str(o).strip()]
    prompt = MATCH_PROMPT.format(task=task, phrase=object_phrase(task),
                                 choices=", ".join(scene) if scene else "(no objects listed)")
    msgs = [{"role": "user", "content": [{"type": "image", "image": initial},
                                         {"type": "text", "text": prompt}]}]
    out = parse_json_object(generate(model, processor, msgs, max_new_tokens=192,
                                     temperature=temperature)) or {}
    return {"object": str(out.get("object", "")).strip().lower(),
            "reason": out.get("reason", ""), "scene_objects": scene}


DIRECT_PROMPT = (
    'Command: "{task}"\n'
    "Which object in the image is this command referring to? The command may not use "
    "the object's exact name — match its term to the object in the image that is "
    "closest in meaning.\n"
    'Return ONLY JSON: {{"object": "<name of the object>", '
    '"appearance": "<a few words on what it looks like>"}}')


def describe_task_object(model, processor, initial: Image.Image, task: str,
                         temperature: float = 0.1) -> dict:
    """Direct mode: ground the task's object with no object list, in one call."""
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": initial},
        {"type": "text", "text": DIRECT_PROMPT.format(task=task)}]}]
    out = parse_json_object(generate(model, processor, msgs, max_new_tokens=192,
                                     temperature=temperature)) or {}
    return {"object": str(out.get("object", "")).strip().lower(),
            "appearance": str(out.get("appearance", ""))}


# Open-vocabulary grasp verification (grasp_verify_match="open"): no object list anywhere.
# The target comes from describe_task_object on the initial scene; the wrist frames are
# described freely; one text-only call decides whether they show the target.
HELD_PROBE_OPEN_PROMPT = (
    "This is a close-up from a camera mounted on a robot gripper; the two orange "
    "fingers are the gripper jaws. Which object is held between the jaws? Answer with "
    "its name, or nothing if no object is held. "
    'Return ONLY JSON: {"object": "<name of the object, or nothing>", '
    '"appearance": "<a few words on what it looks like>"}')

OPEN_VERIFY_PROMPT = (
    'Target object (identified in the scene for the command "{task}"): "{t_name}" — '
    "{t_app}.\n"
    "Wrist camera at the grasp, frame 1: \"{a_name}\" — {a_app}.\n"
    "Wrist camera at the grasp, frame 2: \"{b_name}\" — {b_app}.\n"
    "The wrist camera is very close to the object, so it often shows only part of it (a "
    "surface, a zipper, a texture) and may name that part; judge whether what it shows "
    "is consistent with the target object, not whether the names match.\n"
    "Is the target object the one held in the gripper? Answer yes if both wrist frames "
    "describe the same physical object as the target; no if both wrist frames describe "
    "the same object and it is a different object from the target; unclear if either "
    "frame shows nothing or the two frames describe different objects.\n"
    'Return ONLY JSON: {{"answer": "yes|no|unclear", "reason": "<why>"}}')


def verify_held_open(model, processor, source: ClipSource, close_idx: int,
                     temperature: float) -> list[dict]:
    """Free-text 'what is held' on the wrist frames at close_idx and close_idx + 1."""
    answers = []
    for i in (close_idx, close_idx + 1):
        im = source.wrist[i] if source.wrist and i < len(source.wrist) else None
        if im is None:
            answers.append({"frame": i, "object": "nothing", "appearance": "no wrist frame"})
            continue
        msgs = [{"role": "user", "content": [{"type": "image", "image": im},
                                             {"type": "text", "text": HELD_PROBE_OPEN_PROMPT}]}]
        p = parse_json_object(generate(model, processor, msgs, max_new_tokens=96,
                                       temperature=temperature)) or {}
        answers.append({"frame": i, "object": str(p.get("object", "nothing")).strip().lower(),
                        "appearance": str(p.get("appearance", ""))})
    return answers


def open_verify_decision(model, processor, task: str, target: dict, answers: list[dict],
                         temperature: float) -> tuple[str, str]:
    """yes / no / unclear: do the free-text wrist answers show the grounded target?"""
    if any(a["object"] in ("nothing", "", "none") for a in answers):
        return "unclear", "a wrist frame shows nothing"
    a, b = answers
    text = OPEN_VERIFY_PROMPT.format(task=task, t_name=target.get("object", "?"),
                                     t_app=target.get("appearance", ""),
                                     a_name=a["object"], a_app=a["appearance"],
                                     b_name=b["object"], b_app=b["appearance"])
    msgs = [{"role": "system", "content": [{"type": "text", "text": COMPARE_SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": text}]}]
    out = parse_json_object(generate(model, processor, msgs, max_new_tokens=128,
                                     temperature=temperature)) or {}
    ans = str(out.get("answer", "unclear")).strip().lower()
    return (ans if ans in ("yes", "no", "unclear") else "unclear"), str(out.get("reason", ""))


def verify_held(model, processor, source: ClipSource, close_idx: int,
                temperature: float,
                choices: "list[dict] | None" = None) -> tuple[str, list[dict]]:
    """Wrist-only forced-choice 'what is held' probe on frames close_idx and
    close_idx + 1. Returns (object, answers): the object both frames agree on, or
    'unclear' when either says nothing, they disagree, or a frame is missing."""
    names = ([c["name"] for c in choices] if choices is not None
             else list(HELD_PROBE_OBJECTS))
    shown = (", ".join(f"{c['name']} ({c['appearance']})" if c.get("appearance") else c["name"]
                       for c in choices) if choices is not None else ", ".join(names))
    if choices is not None:
        # A scene-built vocabulary can miss objects; let the wrist say so rather than
        # forcing the held object onto the nearest listed name ('other' never matches
        # the grounded target, so the STOP stands).
        shown += ", other (an object not in this list)"
    prompt = HELD_PROBE_PROMPT.format(choices=shown)
    answers = []
    for i in (close_idx, close_idx + 1):
        im = source.wrist[i] if source.wrist and i < len(source.wrist) else None
        if im is None:
            answers.append({"frame": i, "object": "nothing", "reason": "no wrist frame"})
            continue
        msgs = [{"role": "user", "content": [{"type": "image", "image": im},
                                             {"type": "text", "text": prompt}]}]
        p = parse_json_object(generate(model, processor, msgs, max_new_tokens=128,
                                       temperature=temperature)) or {}
        raw = str(p.get("object", "nothing")).strip().lower()
        obj = raw if raw in ("nothing", "other") else (match_name(raw, names) or raw)
        answers.append({"frame": i, "object": obj, "reason": p.get("reason", "")})
    a0, a1 = answers[0]["object"], answers[1]["object"]
    return (a0 if a0 == a1 and a0 != "nothing" else "unclear"), answers


def approach_gate_windows(source: ClipSource, starts: list[int], win: int,
                          margin: float, tail: int, trigger: str = "open",
                          horizon: int = DEFAULT_APPROACH_GATE_HORIZON,
                          fallback: int = DEFAULT_GRASP_FALLBACK_FRAMES
                          ) -> "list[tuple[bool, str]] | None":
    """Approach-gate decision per window start (see DEFAULT_APPROACH_GATE_MARGIN):
    (open, reason), keyed on the window's LATEST frame. Opens at the trigger — 'open':
    the first frame whose queued or observed gripper aperture is >= `margin` above the
    episode start; 'close': the first frame whose queue plans the grasp close within
    `horizon` steps (see DEFAULT_APPROACH_GATE_HORIZON) — and stays open until `tail`
    frames after the first grasp close. None without joint or plan data."""
    if source.joints is None or source.plans is None or source.initial_joints is None:
        return None
    start_ap = source.initial_joints[_GRIPPER_IDX]
    close = first_grasp_close(source)
    if trigger == "grasp":
        # One window only: ending at the first observed close (look-back over the reach).
        n = len(source.clip)
        if close is not None:
            target, why = close, f"grasp at close@{close} — single look-back check"
        else:
            gs = [j[_GRIPPER_IDX] if j is not None else None for j in source.joints]
            opened = next((i for i, g in enumerate(gs)
                           if g is not None and g - start_ap >= GRIPPER_OPEN_MIN), None)
            target = n - 1 if opened is None else min(n - 1, opened + fallback)
            why = (f"no grasp observed — fallback check @{target} "
                   f"({'gripper never opened' if opened is None else f'opened @{opened} + {fallback}'})")
        target = max(target, win - 1)
        return [((start + win - 1) == target,
                 why if (start + win - 1) == target else
                 ("waiting for the grasp" if start + win - 1 < target else "past the grasp check"))
                for start in starts]
    opened_at = None
    if trigger == "close":
        peak = None
        for i, (j, plan) in enumerate(zip(source.joints, source.plans)):
            if close is not None and i >= close:
                break
            cur = j[_GRIPPER_IDX] if j is not None else None
            if cur is None:
                continue
            peak = cur if peak is None else max(peak, cur)
            ahead = (plan or [])[:horizon]
            if (peak - start_ap >= GRIPPER_OPEN_MIN and ahead
                    and cur - min(ahead) >= DEFAULT_GRASP_CLOSE_MARGIN):
                opened_at = i
                break
        if opened_at is None:
            opened_at = close  # the close was never seen within the horizon at a check
    else:
        for i, (j, plan) in enumerate(zip(source.joints, source.plans)):
            cur = j[_GRIPPER_IDX] if j is not None else None
            top = max([*(plan or []), *([cur] if cur is not None else [])], default=None)
            if top is not None and top - start_ap >= margin:
                opened_at = i
                break
    last_open = (close + tail) if close is not None else len(source.clip) - 1
    out = []
    for start in starts:
        last = start + win - 1
        if opened_at is None or last < opened_at:
            out.append((False, "approach not started (no gripper opening queued yet)"
                        if trigger == "open" else
                        f"no grasp close queued within {horizon} steps yet"))
        elif last > last_open:
            out.append((False, f"past the grasp (close@{close} + {tail} frames)"))
        else:
            out.append((True, f"approach open ({trigger} trigger @{opened_at}, "
                              f"close@{close}, open until @{last_open})"))
    return out


def _grasp_confirmed(verdict: dict) -> bool:
    """True once a window shows the CORRECT object grasped: compare said CONTINUE with
    an explicit match AND the identify pass saw the object held. This is the moment the
    arm has secured the right thing, so the scan COMMITS and retires the semantic check.
    Motion-gated windows (held '?', match 'unclear') and any STOP never satisfy it.
    """
    if not verdict:
        return False
    held = str(verdict.get("held", "")).strip().lower()
    match = str(verdict.get("match", "")).strip().lower()
    action = str(verdict.get("action", "")).strip().upper()
    return action == "CONTINUE" and held in ("yes", "true") and match == "yes"


def run_scan(model, processor, prompts, source: ClipSource, sub_task,
             temperature, fps=2.0, win=4, step=1, stop_on_stop=False,
             motion_thresh=DEFAULT_MOTION_THRESH,
             start_thresh=DEFAULT_JOINT_START_THRESH,
             still_thresh=DEFAULT_JOINT_STILL_THRESH,
             grasp_memory=True,
             grasp_close_margin=DEFAULT_GRASP_CLOSE_MARGIN,
             grasp_imminent_margin=DEFAULT_GRASP_IMMINENT_MARGIN,
             grasp_verify=False, grasp_verify_match="scene",
             approach_gate_margin=DEFAULT_APPROACH_GATE_MARGIN,
             approach_gate_tail=DEFAULT_APPROACH_GATE_TAIL,
             approach_gate_trigger="open",
             approach_gate_horizon=DEFAULT_APPROACH_GATE_HORIZON,
             ground_compare=False,
             grasp_fallback_frames=DEFAULT_GRASP_FALLBACK_FRAMES,
             unclear_wrist_decides=False):
    """Slide a `win`-frame window across the WHOLE run, one two-pass check per step.

    Runs the two-pass identify->compare on every window and prints a per-window
    verdict plus a final timeline. STOP is acted on as soon as it appears (with
    `stop_on_stop`, the scan halts there, mimicking production preemption).

    GRASP COMMIT (`grasp_memory`, default on): once a window confirms the RIGHT
    object has been grasped (`_grasp_confirmed` AND the gripper has actually closed on
    it — the aperture has come in from its open peak by `grasp_close_margin`, skipped
    when a run has no joints), the monitor COMMITS to that object and retires the
    semantic check for the rest of the run — every later window is CONTINUE without
    calling the VLM. The object is in hand and the task is now placement, so a later
    "wrong object" read is only a tray-overlap distractor. This can't mask a real
    wrong grasp: grabbing the WRONG object gives match=no and STOPs BEFORE commit ever
    latches (commit needs match=yes). Pass grasp_memory=False to disable the commit
    and run the two-pass on every window (ablation).

    GRASP-IMMINENT gate (`grasp_imminent_margin` > 0, default off): the VLM is asked
    only while the queued action chunk plans to close the still-open gripper on its
    first grasp (see DEFAULT_GRASP_IMMINENT_MARGIN); every other window is GATED. It
    replaces the start/still/pixel motion gate. Falls back to that gate when the run
    has no joint or plan data.

    GRASP-VERIFY (`grasp_verify`, default off): a STOP is confirmed on the grasp
    itself before it is acted on (see HELD_PROBE_OBJECTS). Vetoed STOPs are logged as
    'VETOED' and commit the grasp. Offline the probe always uses the first-close frames
    of the run (the live system would pause at the close and look then).
    """
    n = len(source.clip)
    if n < win:
        print(f"[ERROR] Run has {n} frame(s); need at least win={win}.")
        return []

    starts = list(range(0, n - win + 1, step))
    print(f"\n{'#'*60}")
    print(f"SCAN — {len(starts)} window(s) of {win} frames (step {step}) over "
          f"{n} frames")
    print(f"Sub-task: \"{sub_task}\"  |  "
          f"{'stop-on-STOP' if stop_on_stop else 'full pass'}  |  "
          f"grasp commit {'on' if grasp_memory else 'off'}")
    imminent = (grasp_imminent_windows(source, starts, win, grasp_imminent_margin)
                if grasp_imminent_margin > 0 else None)
    if imminent is None and (approach_gate_margin > 0
                             or approach_gate_trigger in ("close", "grasp")):
        imminent = approach_gate_windows(source, starts, win, approach_gate_margin,
                                         approach_gate_tail, approach_gate_trigger,
                                         approach_gate_horizon, grasp_fallback_frames)
        detail = (f"close within {approach_gate_horizon} steps" if approach_gate_trigger == "close"
                  else "single check at the grasp" if approach_gate_trigger == "grasp"
                  else f"opening margin {approach_gate_margin:g}")
        print("Approach gate: " + (f"on ({detail}, tail {approach_gate_tail})"
                                   if imminent is not None else
                                   "requested but no joint/plan data — motion gate used"))
    use_imminent = imminent is not None
    if grasp_imminent_margin > 0:
        print("Grasp-imminent gate: " + (
            f"on (margin {grasp_imminent_margin:g}, open ≥ {GRIPPER_OPEN_MIN:g})"
            if use_imminent else "requested but no joint/plan data — motion gate used"))
    print(f"{'#'*60}")

    verify_close = first_grasp_close(source) if grasp_verify else None
    verify_cache = None   # (held, answers) — probed once per run
    grounded = None       # sub-task object grounded on the initial scene (once per run)
    inventory = None      # scene vocabulary (inventory mode), built once per run
    timeline = []
    committed = False     # confirmed the right object is grasped -> retire the check
    held_object = None    # what the gripper was confirmed to pick up (for logging)
    run_max_aperture = None  # widest gripper aperture seen so far (the approach-open)
    for k, start in enumerate(starts):
        win_src = source.window(start, win)
        cur_ap = current_gripper(win_src.joints)
        if cur_ap is not None:
            run_max_aperture = (cur_ap if run_max_aperture is None
                                else max(run_max_aperture, cur_ap))
        # Kinematic grasp test (self-calibrating, object-independent): the gripper has
        # CLOSED on an object once its aperture has come in from the run's open peak by
        # at least the margin. This is authoritative for `held` (passed to run_check) —
        # None when there is no aperture data or the gate is disabled, so held then
        # falls back to the VLM's read.
        if cur_ap is not None and run_max_aperture is not None and grasp_close_margin > 0:
            win_gripper_closed = cur_ap <= run_max_aperture - grasp_close_margin
        else:
            win_gripper_closed = None
        idx_lo = re.match(r"(\d+)", Path(win_src.paths[0]).name)
        idx_hi = re.match(r"(\d+)", Path(win_src.paths[-1]).name)
        span = (f"{idx_lo.group(1) if idx_lo else start}.."
                f"{idx_hi.group(1) if idx_hi else start + win - 1}")

        print(f"\n----- window {k + 1}/{len(starts)} (frames {span})"
              f"{' [COMMITTED]' if committed else ''} -----")

        # Post-commit: the right object is in hand; retire the semantic check entirely
        # (no VLM call). Any later 'wrong object' read would be tray-overlap noise.
        if committed:
            print(f"  [commit] holding '{held_object}' — semantic check retired, "
                  "CONTINUE.")
            timeline.append((span, "CONTINUE", f"committed to '{held_object}'", "held"))
            continue

        gates = dict(motion_thresh=motion_thresh, start_thresh=start_thresh,
                     still_thresh=still_thresh)
        if use_imminent:
            is_open, why = imminent[k]
            print(f"  [{'imminent' if grasp_imminent_margin > 0 else 'approach'} gate] {why} — "
                  + ("asking the VLM." if is_open else "skipping VLM, CONTINUE."))
            if not is_open:
                timeline.append((span, "GATED", f"queue-gated: {why}", "reach"))
                continue
            # The queue already says the arm is reaching to grasp; the motion gate
            # would only block this window (e.g. near-start distance), so disable it.
            gates = dict(motion_thresh=0, start_thresh=0, still_thresh=0)

        # GROUND-COMPARE: resolve the sub-task's object on the initial scene once, at the
        # first reach check, and give pass 2 that object (the verify reuses it).
        target_object = None
        if ground_compare and grasp_verify_match in ("scene", "inventory") \
                and source.initial is not None:
            if grasp_verify_match == "inventory" and inventory is None:
                inventory = scene_inventory(model, processor, source.initial, temperature)
                print("  [ground] scene inventory: "
                      + "; ".join(f"{o['name']} ({o['appearance']})" for o in inventory))
            g_vocab = ([o["name"] for o in inventory] if grasp_verify_match == "inventory"
                       else list(HELD_PROBE_OBJECTS))
            if grounded is None and g_vocab:
                raw = match_task_object(model, processor, source.initial, sub_task,
                                        objects=g_vocab, temperature=temperature)["object"]
                grounded = raw if raw in ("none", "ambiguous") else (match_name(raw, g_vocab) or raw)
                print(f"  [ground] sub-task object grounded on initial scene: '{grounded}'")
            if grounded in g_vocab:
                target_object = grounded

        verdict = run_check(model, processor, prompts, win_src, sub_task,
                            temperature, fps=fps, gripper_closed=win_gripper_closed,
                            target_object=target_object, **gates)
        raw_action = str(verdict.get("action", "?")).upper() if verdict else "PARSE_FAIL"
        gated = bool(verdict) and "motion" in verdict and \
            str(verdict.get("reaching_for", "")).startswith("none (arm not moving)")
        action = "GATED" if gated else raw_action

        # UNCLEAR -> WRIST DECIDES (grasp trigger only): the single look-back check at the
        # grasp often cannot resolve the reach (jaws over the object, neighbours in view).
        # Then the wrist probe on the close frames decides: it clearly shows a scene object
        # that is NOT the grounded target -> STOP (the verify below then confirms it from the
        # same cached answers); target / blind / inconsistent -> CONTINUE as before.
        if (unclear_wrist_decides and approach_gate_trigger == "grasp" and use_imminent
                and grasp_verify and action == "CONTINUE" and verify_close is not None
                and grasp_verify_match in ("scene", "inventory") and source.initial is not None
                and str(verdict.get("reaching_for", "")).strip().lower() in _UNDECIDED_REACH):
            if grasp_verify_match == "inventory" and inventory is None:
                inventory = scene_inventory(model, processor, source.initial, temperature)
            u_vocab = ([o["name"] for o in inventory] if grasp_verify_match == "inventory"
                       else list(HELD_PROBE_OBJECTS))
            if verify_cache is None:
                verify_cache = verify_held(
                    model, processor, source, verify_close, temperature,
                    choices=inventory if grasp_verify_match == "inventory" else None)
            u_held, u_answers = verify_cache
            if grounded is None and u_vocab:
                raw = match_task_object(model, processor, source.initial, sub_task,
                                        objects=u_vocab, temperature=temperature)["object"]
                grounded = raw if raw in ("none", "ambiguous") else (match_name(raw, u_vocab) or raw)
            u_seen = "/".join(a["object"] for a in u_answers)
            if u_held in u_vocab and grounded in u_vocab and u_held != grounded:
                action = "STOP"
                verdict["action"] = "STOP"
                verdict["reason"] = (f"[wrist-decided: reach unclear; wrist held '{u_held}' "
                                     f"({u_seen}), task object '{grounded}'] "
                                     f"{verdict.get('reason', '')}")
                print(f"  [unclear->wrist] reach unclear; wrist holds '{u_held}' ({u_seen}) "
                      f"!= task object '{grounded}' — STOP")
            else:
                verdict["reason"] = (f"[wrist: reach unclear; wrist '{u_held}' ({u_seen}), "
                                     f"task object '{grounded}' — CONTINUE] "
                                     f"{verdict.get('reason', '')}")
                print(f"  [unclear->wrist] reach unclear; wrist '{u_held}' ({u_seen}) — CONTINUE")

        if grasp_verify and action == "STOP" and grasp_verify_match == "open":
            if verify_close is None or source.initial is None:
                note = "no observed grasp close to verify on"
            else:
                if grounded is None:
                    grounded = describe_task_object(model, processor, source.initial,
                                                    sub_task, temperature)
                    print(f"  [grasp verify] open grounding on initial scene: "
                          f"'{grounded['object']}' [{grounded['appearance']}]")
                if verify_cache is None:
                    answers = verify_held_open(model, processor, source, verify_close,
                                               temperature)
                    verify_cache = (open_verify_decision(model, processor, sub_task,
                                                         grounded, answers, temperature),
                                    answers)
                (ans, why), answers = verify_cache
                seen = " / ".join(f"{a['object']} [{a['appearance']}]" for a in answers)
                if ans == "yes":
                    print(f"  [grasp verify] wrist at close@{verify_close}: {seen} = target "
                          f"'{grounded['object']}' — STOP VETOED, committing.")
                    timeline.append((span, "VETOED",
                                     f"vetoed: wrist {seen} = target '{grounded['object']}' "
                                     f"[{grounded['appearance']}]; reach said: "
                                     f"{verdict.get('reason', '')}", "reach"))
                    committed, held_object = True, answers[0]["object"]
                    continue
                note = (f"open verify {ans}: wrist {seen}; target '{grounded['object']}' "
                        f"[{grounded['appearance']}] — {why}")
            print(f"  [grasp verify] STOP stands — {note}.")
            verdict["reason"] = f"[verify: {note}] {verdict.get('reason', '')}"
        elif grasp_verify and action == "STOP":
            if verify_close is None:
                note = "no observed grasp close to verify on"
            else:
                if grasp_verify_match == "inventory" and inventory is None:
                    inventory = (scene_inventory(model, processor, source.initial, temperature)
                                 if source.initial is not None else [])
                    print(f"  [grasp verify] scene inventory: "
                          + "; ".join(f"{o['name']} ({o['appearance']})" for o in inventory))
                vocab = ([o["name"] for o in inventory] if grasp_verify_match == "inventory"
                         else list(HELD_PROBE_OBJECTS))
                if verify_cache is None:
                    verify_cache = verify_held(
                        model, processor, source, verify_close, temperature,
                        choices=inventory if grasp_verify_match == "inventory" else None)
                held, answers = verify_cache
                seen = "/".join(a["object"] for a in answers)
                if grasp_verify_match in ("scene", "inventory") and grounded is None:
                    raw = (match_task_object(model, processor, source.initial, sub_task,
                                             objects=vocab, temperature=temperature)["object"]
                           if source.initial is not None and vocab else "")
                    grounded = raw if raw in ("none", "ambiguous") else (match_name(raw, vocab) or raw)
                    print(f"  [grasp verify] sub-task object grounded on initial scene: "
                          f"'{grounded}'")
                scene_match = (grasp_verify_match in ("scene", "inventory")
                               and grounded in (*vocab, "none"))
                if held == "unclear":
                    note = f"wrist blind or inconsistent at close@{verify_close} ({seen})"
                elif scene_match:
                    if held == grounded:
                        print(f"  [grasp verify] wrist at close@{verify_close} holds "
                              f"'{held}' ({seen}) = grounded target — STOP VETOED, committing.")
                        timeline.append((span, "VETOED",
                                         f"vetoed: wrist held '{held}' ({seen}) = task object "
                                         f"'{grounded}'; reach said: {verdict.get('reason', '')}",
                                         "reach"))
                        committed, held_object = True, held
                        continue
                    note = (f"wrist held '{held}' ({seen}), task object grounded to "
                            f"'{grounded}'")
                else:
                    cmp_msgs = [
                        {"role": "system", "content": [{"type": "text", "text": prompts["compare"]}]},
                        {"role": "user", "content": _compare_content(
                            sub_task, {"reaching_for": held, "candidates": [held], "held": "yes"})}]
                    cv = parse_json_object(generate(
                        model, processor, cmp_msgs, max_new_tokens=MONITOR_MAX_NEW_TOKENS,
                        temperature=temperature)) or {}
                    if str(cv.get("action", "")).upper() == "CONTINUE":
                        print(f"  [grasp verify] wrist at close@{verify_close} holds "
                              f"'{held}' ({seen}) = target — STOP VETOED, committing.")
                        timeline.append((span, "VETOED",
                                         f"vetoed: wrist held '{held}' ({seen}); reach said: "
                                         f"{verdict.get('reason', '')}", "reach"))
                        committed, held_object = True, held
                        continue
                    note = f"wrist held '{held}' ({seen}), not the target"
            print(f"  [grasp verify] STOP stands — {note}.")
            verdict["reason"] = f"[verify: {note}] {verdict.get('reason', '')}"

        timeline.append((span, action, verdict.get("reason", "") if verdict else "",
                         "reach"))
        if stop_on_stop and action == "STOP":
            print(f"\n[SCAN] STOP at window {k + 1} (frames {span}); halting.")
            break

        # Commit latch: the reach pass confirmed the RIGHT object AND the gripper has
        # actually closed on it. `held` is already kinematic (win_gripper_closed was
        # passed to run_check), so _grasp_confirmed (which needs held=yes) can only be
        # true here when the gripper is closed or there was no aperture data
        # (win_gripper_closed is None -> VLM-only commit). From here the check is retired.
        if grasp_memory and _grasp_confirmed(verdict):
            if win_gripper_closed is False:
                print(f"  [grasp gate] reach says grasped, but gripper not yet closed "
                      f"from its open peak (aperture {_fmt(cur_ap)} vs peak "
                      f"{_fmt(run_max_aperture)}, need ≤ peak−{grasp_close_margin:g}) — "
                      "not committing yet.")
            else:
                held_object = str(verdict.get("reaching_for") or "").strip() or "object"
                committed = True
                print(f"\n[SCAN] grasp confirmed at window {k + 1} (frames {span}); "
                      f"committing to '{held_object}' (gripper {_fmt(cur_ap)}, peak "
                      f"{_fmt(run_max_aperture)}) — semantic check retired for the rest "
                      "of the run.")

    # --- Timeline summary ---
    print(f"\n{'#'*60}\nSCAN TIMELINE\n{'#'*60}")
    for span, action, reason, phase in timeline:
        line = f"  frames {span:<9} [{phase:<5}] {action}"
        if action == "STOP" and reason:
            line += f"  — {reason}"
        print(line)
    print(f"\n  {len(timeline)} window(s): "
          f"{sum(1 for t in timeline if t[1] == 'STOP')} STOP, "
          f"{sum(1 for t in timeline if t[1] == 'CONTINUE')} CONTINUE, "
          f"{sum(1 for t in timeline if t[1] == 'GATED')} motion-gated, "
          f"{sum(1 for t in timeline if t[1] == 'PARSE_FAIL')} parse-fail")
    return timeline


# ---------------------------------------------------------------------------
# Prompt file / editor helpers
# ---------------------------------------------------------------------------

# A --prompt-file bundles both prompts in one text file, split by these markers,
# so /reload can refresh them without touching the model.
_PROMPT_MARKERS = {
    "identify": "### IDENTIFY ###",
    "compare": "### COMPARE ###",
}


def serialize_prompts(prompts: dict) -> str:
    return "\n".join(
        f"{_PROMPT_MARKERS[key]}\n{prompts[key].strip()}\n"
        for key in ("identify", "compare")
    )


def parse_prompt_file(text: str, base: dict) -> dict:
    """Parse a prompt file into {identify, compare}.

    Sections are delimited by the markers above. Any section missing from the
    file keeps its value from `base`, so a file may override just one prompt.
    """
    result = dict(base)
    positions = []
    for key, marker in _PROMPT_MARKERS.items():
        idx = text.find(marker)
        if idx != -1:
            positions.append((idx, key, marker))
    positions.sort()
    for i, (idx, key, marker) in enumerate(positions):
        start = idx + len(marker)
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        result[key] = text[start:end].strip()
    return result


def edit_in_editor(initial_text: str, suffix: str = ".txt") -> str | None:
    """Open $EDITOR on initial_text; return the edited text, or None on failure."""
    editor = os.environ.get("EDITOR", "vi")
    with tempfile.NamedTemporaryFile("w+", suffix=suffix, delete=False) as f:
        f.write(initial_text)
        path = f.name
    try:
        subprocess.run([editor, path], check=True)
        return Path(path).read_text()
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] Editor failed: {e}")
        return None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

INTERACTIVE_HELP = """
Commands:
  <any text>          Run the two-pass check on the current clip for this sub-task
                      (e.g. "put the banana on the tray")
  /clip <glob|dir>    Load a whole run of full-res frames. A single check uses the
                      run's first frame as the initial scene + the last {win} as
                      the trajectory; /scan slides across all of them.
                      (current: {source})
  /scan [sub-task]    Slide the {win}-frame window across EVERY frame in the run,
                      one two-pass check per step, then print a verdict timeline. Task
                      defaults to the filename's. Once the right object is confirmed
                      grasped, the check is retired for the rest of the run (commit).
  /window <n>         Set the sliding-window (trajectory) size (current: {win})
  /gclose <value>     Grasp-commit gate (self-calibrating): commit to the grasped
                      object (retire the check) only once the aperture is this far
                      below the widest opening seen so far (fingers closed on the
                      object). 0 disables. (current: {gclose})
  /imminent <value>   Grasp-imminent gate for /scan: ask the VLM only while the
                      queued action chunk plans to close the open gripper by at
                      least this much (first grasp only). Replaces the motion gate.
                      0 disables. (current: {imminent})
  /fps <value>        Clip playback fps for video timestamps (current: {fps})
  /jstart <value>     JOINT startup gate: L2 of the current pose from the EPISODE-START
                      pose; below it the arm is still near home -> gate. 0 disables.
                      (current: {jstart})
  /jstill <value>     JOINT still gate: peak L2 within the window; below it the arm is
                      barely moving now (settled/paused end) -> gate. 0 disables.
                      (current: {jstill})
  /motion <value>     PIXEL-motion gate (fallback when no joint CSV); below it a window
                      is skipped to CONTINUE. 0 disables. (current: {motion})
  /temp <value>       Set temperature (current: {temp})
  /prompt [which]     Print a prompt: identify | compare | all (default all)
  /edit [which]       Edit a prompt live in $EDITOR: identify | compare
  /reload             Reload prompts from --prompt-file (current: {pfile})
  /save <path>        Write the current prompts to a file (reloadable with /reload)
  /help               Show this help
  /quit               Exit
""".strip()


def source_label(source) -> str:
    if source is None:
        return "none"
    first, last = Path(source.paths[0]).name, Path(source.paths[-1]).name
    return f"clip x{len(source.clip)} ({first} .. {last})"


def interactive_loop(model, processor, prompts, source,
                     prompt_file, temp, fps=2.0, clip_window=4,
                     motion_thresh=DEFAULT_MOTION_THRESH,
                     start_thresh=DEFAULT_JOINT_START_THRESH,
                     still_thresh=DEFAULT_JOINT_STILL_THRESH,
                     grasp_close_margin=DEFAULT_GRASP_CLOSE_MARGIN,
                     wrist_rotate=DEFAULT_WRIST_ROTATE_CCW,
                     grasp_imminent_margin=DEFAULT_GRASP_IMMINENT_MARGIN,
                     grasp_verify=False, grasp_verify_match="scene",
                     approach_gate_margin=DEFAULT_APPROACH_GATE_MARGIN,
                     approach_gate_tail=DEFAULT_APPROACH_GATE_TAIL,
                     approach_gate_trigger="open",
                     approach_gate_horizon=DEFAULT_APPROACH_GATE_HORIZON,
                     ground_compare=False,
                     grasp_fallback_frames=DEFAULT_GRASP_FALLBACK_FRAMES,
                     unclear_wrist_decides=False):
    print(f"\n{'='*60}")
    print("MONITOR TUNING MODE — model loaded, type a sub-task to run a check.")
    print(f"Source: {source_label(source)}")
    print("Type /help for commands, /quit to exit.")
    print(f"{'='*60}")

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input in ("/quit", "/exit"):
            print("Exiting.")
            break

        elif user_input == "/help":
            print(INTERACTIVE_HELP.format(
                source=source_label(source),
                win=clip_window,
                fps=fps,
                jstart=start_thresh,
                jstill=still_thresh,
                motion=motion_thresh,
                gclose=grasp_close_margin,
                imminent=grasp_imminent_margin,
                temp=temp,
                pfile=prompt_file or "none",
            ))

        elif user_input.startswith("/clip"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /clip <glob|dir>  (e.g. /clip vla_failure_test/toy/000_toy)")
                continue
            try:
                source = load_clip(parts[1].strip(), wrist_rotate=wrist_rotate)
                print(f"Loaded {source_label(source)}")
                task_guess = task_from_path(source.paths[-1])
                if task_guess:
                    print(f"  (filename suggests task: \"{task_guess}\")")
            except Exception as e:  # noqa: BLE001
                print(f"[ERROR] Could not load clip: {e}")

        elif user_input.startswith("/scan"):
            if not isinstance(source, ClipSource):
                print("[ERROR] /scan needs a clip. Load one with /clip <dir>.")
                continue
            task = user_input[len("/scan"):].strip() or task_from_path(source.paths[-1])
            if not task:
                print("Provide a sub-task: /scan put the toy on the tray")
                continue
            run_scan(model, processor, prompts, source, task, temp,
                     fps=fps, win=clip_window, step=1, motion_thresh=motion_thresh,
                     start_thresh=start_thresh, still_thresh=still_thresh,
                     grasp_close_margin=grasp_close_margin,
                     grasp_imminent_margin=grasp_imminent_margin,
                     grasp_verify=grasp_verify, grasp_verify_match=grasp_verify_match,
                     approach_gate_margin=approach_gate_margin,
                     approach_gate_tail=approach_gate_tail,
                     approach_gate_trigger=approach_gate_trigger,
                     approach_gate_horizon=approach_gate_horizon,
                     ground_compare=ground_compare,
                     grasp_fallback_frames=grasp_fallback_frames,
                     unclear_wrist_decides=unclear_wrist_decides)

        elif user_input.startswith("/imminent"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 1:
                print(f"Grasp-imminent gate is {grasp_imminent_margin} "
                      f"({'off' if grasp_imminent_margin <= 0 else 'on'}). "
                      "Set with /imminent <float> (0 disables).")
                continue
            try:
                grasp_imminent_margin = float(parts[1].strip())
                print(f"Grasp-imminent gate set to {grasp_imminent_margin} "
                      f"({'off' if grasp_imminent_margin <= 0 else 'on'})")
            except ValueError:
                print("Usage: /imminent <float>  (e.g. /imminent 6, or /imminent 0 to disable)")

        elif user_input.startswith("/gclose"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print(f"Grasp-commit gate (close margin below the open peak) is "
                      f"{grasp_close_margin} "
                      f"({'off — VLM-only commit' if grasp_close_margin <= 0 else 'on'}). "
                      "Set with /gclose <float> (0 disables).")
                continue
            try:
                grasp_close_margin = float(parts[1].strip())
                print(f"Grasp-commit gate set to {grasp_close_margin} "
                      f"({'off' if grasp_close_margin <= 0 else 'on'})")
            except ValueError:
                print("Usage: /gclose <float>  (e.g. /gclose 6, or /gclose 0 to disable)")

        elif user_input.startswith("/window"):
            try:
                clip_window = int(user_input.split(maxsplit=1)[1].strip())
                print(f"Window size set to {clip_window}")
            except (ValueError, IndexError):
                print("Usage: /window <int>  (e.g. /window 4)")

        elif user_input.startswith("/fps "):
            try:
                fps = float(user_input[5:].strip())
                print(f"Clip fps set to {fps}")
            except ValueError:
                print("Usage: /fps <float>  (e.g. /fps 2.0)")

        elif user_input.startswith("/jstart"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print(f"Joint startup gate (dist from episode start) is {start_thresh} "
                      f"({'off' if start_thresh <= 0 else 'on'}). "
                      "Set with /jstart <float> (0 disables).")
                continue
            try:
                start_thresh = float(parts[1].strip())
                print(f"Joint startup gate set to {start_thresh} "
                      f"({'off' if start_thresh <= 0 else 'on'})")
            except ValueError:
                print("Usage: /jstart <float>  (e.g. /jstart 15, or /jstart 0 to disable)")

        elif user_input.startswith("/jstill"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print(f"Joint still gate (within-window peak) is {still_thresh} "
                      f"({'off' if still_thresh <= 0 else 'on'}). "
                      "Set with /jstill <float> (0 disables).")
                continue
            try:
                still_thresh = float(parts[1].strip())
                print(f"Joint still gate set to {still_thresh} "
                      f"({'off' if still_thresh <= 0 else 'on'})")
            except ValueError:
                print("Usage: /jstill <float>  (e.g. /jstill 6, or /jstill 0 to disable)")

        elif user_input.startswith("/motion"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print(f"Pixel-motion gate is {motion_thresh} "
                      f"({'off' if motion_thresh <= 0 else 'on'}). "
                      "Set with /motion <float> (0 disables).")
                continue
            try:
                motion_thresh = float(parts[1].strip())
                print(f"Pixel-motion gate set to {motion_thresh} "
                      f"({'off' if motion_thresh <= 0 else 'on'})")
            except ValueError:
                print("Usage: /motion <float>  (e.g. /motion 2.0, or /motion 0 to disable)")

        elif user_input.startswith("/temp "):
            try:
                temp = float(user_input[6:].strip())
                print(f"Temperature set to {temp}")
            except ValueError:
                print("Usage: /temp <float>  (e.g. /temp 0.1)")

        elif user_input.startswith("/prompt"):
            parts = user_input.split(maxsplit=1)
            which = parts[1].strip() if len(parts) > 1 else "all"
            keys = ("identify", "compare") if which == "all" else (which,)
            for key in keys:
                if key not in prompts:
                    print(f"Unknown prompt: {key}. Use identify | compare | all")
                    continue
                print(f"\n{'#'*60}\n# {key.upper()} PROMPT\n{'#'*60}\n{prompts[key]}")

        elif user_input.startswith("/edit"):
            parts = user_input.split(maxsplit=1)
            which = parts[1].strip() if len(parts) > 1 else ""
            if which not in prompts:
                print("Usage: /edit identify | compare")
                continue
            edited = edit_in_editor(prompts[which], suffix=f"_{which}.txt")
            if edited is not None and edited.strip():
                prompts[which] = edited.strip()
                print(f"Updated the {which} prompt ({len(prompts[which])} chars). "
                      "Use /save to persist it to a file.")
            else:
                print("No change (empty or editor failed).")

        elif user_input == "/reload":
            if not prompt_file:
                print("No --prompt-file set. Start with --prompt-file <path>, or "
                      "use /save to create one first.")
                continue
            try:
                text = Path(prompt_file).read_text()
                prompts = parse_prompt_file(text, prompts)
                print(f"Reloaded prompts from {prompt_file}.")
            except Exception as e:  # noqa: BLE001
                print(f"[ERROR] Could not reload: {e}")

        elif user_input.startswith("/save"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /save <path>")
                continue
            path = parts[1].strip()
            try:
                Path(path).write_text(serialize_prompts(prompts))
                if not prompt_file:
                    prompt_file = path  # so /reload targets what we just wrote
                print(f"Saved prompts to {path}. (/reload will read from "
                      f"{prompt_file}.)")
            except Exception as e:  # noqa: BLE001
                print(f"[ERROR] Could not save: {e}")

        elif user_input.startswith("/"):
            print(f"Unknown command: {user_input}. Type /help for options.")

        # --- Run a check ---
        else:
            if source is None:
                print("[ERROR] No clip loaded. Use /clip (or start with --clip).")
                continue
            run_check(model, processor, prompts, source.tail(clip_window),
                      user_input, temp, fps=fps, motion_thresh=motion_thresh,
                      start_thresh=start_thresh, still_thresh=still_thresh)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive harness for tuning the two-pass semantic monitor"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--clip", default=None,
        help="Glob or directory of ordered full-res frames (e.g. "
             "vla_failure_test/toy/000_toy). The first frame is the initial scene; "
             "the last --window frames are the trajectory.",
    )
    parser.add_argument(
        "--view", default="front",
        help="Filename marker for the FRONT (primary) camera track — fixes ordering, "
             "the initial scene, and the front half of the window. Default: front",
    )
    parser.add_argument(
        "--wrist-view", default="wrist",
        help="Filename marker for the WRIST (in-hand) camera track, loaded in parallel "
             "and aligned to the front frames to form the wrist half of the window. "
             "Default: wrist",
    )
    parser.add_argument(
        "--wrist-rotate", type=float, default=DEFAULT_WRIST_ROTATE_CCW,
        help="Degrees COUNTER-CLOCKWISE to rotate each wrist frame at load so it matches "
             "the front view's orientation (the wrist camera is mounted 90° clockwise). "
             "0 leaves wrist frames unrotated; flip the sign if the mounting is CCW. "
             f"Default: {DEFAULT_WRIST_ROTATE_CCW}",
    )
    parser.add_argument(
        "--window", type=int, default=4,
        help="Sliding-window (trajectory) size: frames per check (the tail for a "
             "single check, the window for /scan). Matches vlm_monitor_num_frames. "
             "Default: 4",
    )
    parser.add_argument(
        "--scan", action="store_true",
        help="Slide the window across the whole run on startup (then drop into the "
             "REPL). Task from --task or the filenames.",
    )
    parser.add_argument(
        "--task", default=None,
        help="Sub-task for --scan (default: read from the frame filenames)",
    )
    parser.add_argument(
        "--fps", type=float, default=2.0,
        help="Clip playback fps for video timestamps (default: 2.0)",
    )
    parser.add_argument(
        "--temp", type=float, default=0.1,
        help="Sampling temperature (default: 0.1)",
    )
    parser.add_argument(
        "--joint-start-thresh", type=float, default=DEFAULT_JOINT_START_THRESH,
        help="JOINT startup gate: L2 of the current pose from the EPISODE-START pose "
             "(from actions.csv). Below it the arm is still near home -> gate to "
             f"CONTINUE. 0 disables. Default: {DEFAULT_JOINT_START_THRESH}",
    )
    parser.add_argument(
        "--joint-still-thresh", type=float, default=DEFAULT_JOINT_STILL_THRESH,
        help="JOINT still gate: peak L2 displacement within the window relative to its "
             "first frame. Below it the arm is barely moving now (settled/paused end) "
             f"-> gate to CONTINUE. 0 disables. Default: {DEFAULT_JOINT_STILL_THRESH}",
    )
    parser.add_argument(
        "--motion-thresh", type=float, default=DEFAULT_MOTION_THRESH,
        help="PIXEL-motion gate threshold (mean inter-frame pixel diff, 0-255), used "
             "as the fallback when a run has no joint CSV. Below it a window "
             "short-circuits to CONTINUE without asking the VLM. 0 disables. "
             f"Default: {DEFAULT_MOTION_THRESH}",
    )
    parser.add_argument(
        "--no-grasp-memory", action="store_true",
        help="Disable the grasp COMMIT in --scan (do not retire the semantic check "
             "after the right object is grasped; run the two-pass on every window) — "
             "ablation",
    )
    parser.add_argument(
        "--grasp-close-margin", type=float, default=DEFAULT_GRASP_CLOSE_MARGIN,
        help="Grasp-commit gate (self-calibrating, object-independent): commit to the "
             "grasped object (retiring the semantic check) only once the gripper "
             "aperture is at least this far BELOW the widest opening seen so far this "
             "run — i.e. the fingers have closed in from the approach-open. A reach "
             "'held: yes' while still near the open peak is premature. 0 disables "
             f"(VLM-only commit). Default: {DEFAULT_GRASP_CLOSE_MARGIN}",
    )
    parser.add_argument(
        "--grasp-verify", action="store_true",
        help="Hybrid: confirm each STOP on the grasp itself — a wrist-only 'what is "
             "held' probe at the first gripper close vetoes the STOP when it shows the "
             "sub-task's target (see HELD_PROBE_OBJECTS).",
    )
    parser.add_argument(
        "--approach-gate-margin", type=float, default=DEFAULT_APPROACH_GATE_MARGIN,
        help="Approach gate: ask the VLM only from the queued gripper opening (aperture "
             ">= this above the start) until --approach-gate-tail frames after the first "
             "grasp close; replaces the motion gate. 0 disables. "
             f"Default: {DEFAULT_APPROACH_GATE_MARGIN}",
    )
    parser.add_argument(
        "--approach-gate-tail", type=int, default=DEFAULT_APPROACH_GATE_TAIL,
        help=f"Frames after the first grasp close the approach gate stays open. "
             f"Default: {DEFAULT_APPROACH_GATE_TAIL}",
    )
    parser.add_argument(
        "--approach-gate-trigger", choices=["open", "close", "grasp"], default="open",
        help="Approach gate trigger: 'open' = queued gripper opening (needs "
             "--approach-gate-margin > 0); 'close' = the grasp close queued within "
             "--approach-gate-horizon steps (enables the gate on its own).",
    )
    parser.add_argument(
        "--unclear-wrist-decides", action="store_true",
        help="Grasp trigger: when the single look-back check is 'unclear', the wrist probe "
             "decides (a clearly different scene object -> STOP).",
    )
    parser.add_argument(
        "--grasp-fallback-frames", type=int, default=DEFAULT_GRASP_FALLBACK_FRAMES,
        help="Grasp trigger: if no close is observed, check once this many frames after "
             f"the gripper opened. Default: {DEFAULT_GRASP_FALLBACK_FRAMES}",
    )
    parser.add_argument(
        "--approach-gate-horizon", type=int, default=DEFAULT_APPROACH_GATE_HORIZON,
        help=f"Look-ahead (control steps) for the close trigger. "
             f"Default: {DEFAULT_APPROACH_GATE_HORIZON}",
    )
    parser.add_argument(
        "--ground-compare", action="store_true",
        help="Give pass 2 the sub-task's object grounded on the initial scene (per "
             "--grasp-verify-match: scene list or inventory) instead of the task wording.",
    )
    parser.add_argument(
        "--grasp-verify-match", choices=["scene", "inventory", "text", "open"], default="scene",
        help="How --grasp-verify decides the wrist's object is the target: 'scene' "
             "grounds the sub-task's object on the initial scene and compares objects "
             "(default); 'text' uses the compare pass on the sub-task wording.",
    )
    parser.add_argument(
        "--grasp-imminent-margin", type=float, default=DEFAULT_GRASP_IMMINENT_MARGIN,
        help="Grasp-imminent gate for --scan / /scan: ask the VLM only while the "
             "queued action chunk (act_gripper in actions.csv) plans to close the "
             "still-open gripper by at least this much, on the first grasp only. "
             "Replaces the start/still/pixel motion gate. 0 disables. "
             f"Default: {DEFAULT_GRASP_IMMINENT_MARGIN}",
    )
    parser.add_argument(
        "--prompt-file", default=None,
        help="Load prompt overrides from this file (editable, reloadable with /reload)",
    )
    args = parser.parse_args()

    # --- Resolve prompts ---
    prompts = default_prompts()
    prompt_file = args.prompt_file
    if prompt_file:
        p = Path(prompt_file)
        if p.exists():
            prompts = parse_prompt_file(p.read_text(), prompts)
            print(f"Loaded prompt overrides from {prompt_file}.")
        else:
            print(f"[INFO] --prompt-file {prompt_file} does not exist yet; using "
                  "built-in prompts. /save will create it.")

    # --- Resolve source ---
    source = None
    if args.clip:
        try:
            source = load_clip(args.clip, view=args.view, wrist_view=args.wrist_view,
                               wrist_rotate=args.wrist_rotate)
            print(f"Using {source_label(source)}")
            task_guess = task_from_path(source.paths[-1])
            if task_guess:
                print(f"  (filename suggests task: \"{task_guess}\")")
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] Could not load clip: {e}")
            return
    else:
        print("[INFO] No --clip given; set one with /clip before running a check.")

    # --- Load model ---
    model, processor = load_model(args.model)

    # --- Optional one-shot scan on startup ---
    if args.scan:
        if not isinstance(source, ClipSource):
            print("[ERROR] --scan needs --clip.")
            return
        task = args.task or task_from_path(source.paths[-1])
        if not task:
            print("[ERROR] --scan needs a task; pass --task or use frames whose "
                  "names carry it (put_the_X_on_the_tray).")
        else:
            run_scan(model, processor, prompts, source, task, args.temp,
                     fps=args.fps, win=args.window, step=1,
                     motion_thresh=args.motion_thresh,
                     start_thresh=args.joint_start_thresh,
                     still_thresh=args.joint_still_thresh,
                     grasp_memory=not args.no_grasp_memory,
                     grasp_close_margin=args.grasp_close_margin,
                     grasp_imminent_margin=args.grasp_imminent_margin,
                     grasp_verify=args.grasp_verify,
                     grasp_verify_match=args.grasp_verify_match,
                     approach_gate_margin=args.approach_gate_margin,
                     approach_gate_tail=args.approach_gate_tail,
                     approach_gate_trigger=args.approach_gate_trigger,
                     approach_gate_horizon=args.approach_gate_horizon,
                     ground_compare=args.ground_compare,
                     grasp_fallback_frames=args.grasp_fallback_frames,
                     unclear_wrist_decides=args.unclear_wrist_decides)

    interactive_loop(model, processor, prompts, source,
                     prompt_file, args.temp, fps=args.fps, clip_window=args.window,
                     motion_thresh=args.motion_thresh,
                     start_thresh=args.joint_start_thresh,
                     still_thresh=args.joint_still_thresh,
                     grasp_close_margin=args.grasp_close_margin,
                     wrist_rotate=args.wrist_rotate,
                     grasp_imminent_margin=args.grasp_imminent_margin,
                     grasp_verify=args.grasp_verify,
                     grasp_verify_match=args.grasp_verify_match,
                     approach_gate_margin=args.approach_gate_margin,
                     approach_gate_tail=args.approach_gate_tail,
                     approach_gate_trigger=args.approach_gate_trigger,
                     approach_gate_horizon=args.approach_gate_horizon,
                     ground_compare=args.ground_compare,
                     grasp_fallback_frames=args.grasp_fallback_frames,
                     unclear_wrist_decides=args.unclear_wrist_decides)


if __name__ == "__main__":
    main()
