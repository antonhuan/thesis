"""
Interactive harness for tuning the Qwen3-VL continuous monitor.

The monitor is the online fault-catcher of the dual-system architecture: while
the VLA executes a sub-task, the monitor watches the recent trajectory and calls
CONTINUE (progressing legally through the phase arc) or STOP (a clear
expectation-violation, e.g. a dropped grasp). It lives in
vlm_robot_orchestrator.py as VLMPlanner.monitor() + MONITOR_SYSTEM_PROMPT.

This script is a focused REPL for iterating on that monitor ALONE — load the
model once, point it at a saved frame, type the sub-task, and inspect the
verdict. Iterate on the system prompt (/edit) and the pass structure (/split)
without re-running the robot, then paste the winning prompt back into the
orchestrator.

Two knobs, matching what actually changes the monitor's behaviour:

  1. Pass structure (/split):
       single-call — one call emits observations + verdict together (a faithful
                     copy of the orchestrator's MONITOR_SYSTEM_PROMPT).
       observe->judge — pass 1 describes the current frame (observations only),
                     pass 2 takes that description and returns the verdict. Same
                     split the decomposition and final-eval paths already use.

  2. System prompt (/edit, /prompt, /reload) — edit the live prompt in $EDITOR,
     print it, or reload it from --prompt-file, all without dropping the model.

Image input: the orchestrator saves each monitor look as a contact-sheet grid
(monitor_*.png under runs/<run>/<task>/) — N frames tiled in time order, each
tile labelled t=..s #i. This harness feeds that grid as a SINGLE still: the
model sees every tile at once, and the last tile (#n) is the current frame.
This differs from the production path (a true video clip + a separate
full-detail current still), so treat verdicts here as a prompt-tuning signal,
not a bit-exact replay.

Shared model loading and inference live in vlm_core.py.

Requirements:
    pip install torch transformers accelerate pillow numpy
    pip install git+https://github.com/huggingface/transformers   # Qwen3-VL

Usage:
    # Tune against one saved monitor grid:
    python vlm_monitor_tune.py --image runs/run_.../01/monitor_..._chk002_CONTINUE.png

    # Start in observe->judge mode:
    python vlm_monitor_tune.py --image grid.png --split

    # Load prompt overrides from a file you can edit + /reload:
    python vlm_monitor_tune.py --image grid.png --prompt-file my_monitor_prompt.txt
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from vlm_core import load_model, generate

# Number of tokens the monitor is allowed to generate. Matches the orchestrator's
# VLMPlanner.monitor() call so verdict length here tracks production.
MONITOR_MAX_NEW_TOKENS = 384


# ---------------------------------------------------------------------------
# Monitor prompts
#
# MONITOR_SYSTEM_PROMPT is a verbatim copy of the orchestrator's prompt so that
# whatever you tune here can be pasted straight back into
# vlm_robot_orchestrator.py. The OBSERVE/JUDGE pair is this harness's split of
# that same prompt along its own "OBSERVE FIRST, LABEL AFTER" seam — pass 1 only
# describes, pass 2 only judges given the description.
# ---------------------------------------------------------------------------

MONITOR_SYSTEM_PROMPT = """You are a real-time robot trajectory monitor. You watch a robot arm execute a sub-task and catch it going wrong *before* it finishes on a bad premise. You are NOT the success judge — deciding "did it succeed" is a separate module that fires at the end. Never call the task complete; your only job is "is this going wrong?".

WHAT YOU RECEIVE EACH LOOK
- The sub-task the arm is attempting.
- The expected PHASE ARC for this kind of sub-task: the ordered stages a correct
  attempt passes through. Anomalies are illegal jumps in this arc, not vague
  "looks off" impressions.
- A short video clip of the last few seconds. Its frames are in time order:
  Frame 1 is the EARLIEST, the final frame is the LATEST. Read the direction of
  motion across the frames — do not treat them as an unordered set.
- The CURRENT FRAME on its own, in full detail: the single still that shows the
  scene RIGHT NOW (it is the same moment as the clip's last frame, at higher
  resolution).
- A PRIOR ASSESSMENT from your previous look (the phase/status you reported last
  time). Your last look was the moment just before Frame 1, so it stitches
  directly onto the first frame of this clip.

GROUND YOUR DESCRIPTION IN THE CURRENT FRAME
Read `phase`, `object_status` and `gripper_status` off the CURRENT FRAME — it is
the true present state. Do not describe an averaged or earlier moment from the
clip. Use the video ONLY to judge the direction of motion and the transition since
your prior look. Where the current frame is clear, prefer it; name only what you
can actually see in it (object positions, whether the gripper holds anything),
never what the sub-task implies should be there.

THE PRIOR IS A HYPOTHESIS TO CHECK, NOT GROUND TRUTH
Verify it against the frames. If the frames contradict it, believe the frames and
say so — a grasp you reported "complete" but that actually slipped is exactly the
fault you exist to catch. If you cannot contradict the prior, threading it makes
you blind.

VERDICT — binary
- "CONTINUE": the attempt is progressing legally through the arc, OR you are
  unsure. Default here.
- "STOP" (fault): a CLEAR expectation-violation — an illegal transition or
  sustained wrong-direction motion. STOP preempts the arm, so a spurious STOP
  aborts a good run. Raise it ONLY on a clear violation, never on momentary
  ambiguity or a single flickery frame. Prefer to miss a marginal case and let
  the end-of-task judge catch it over aborting a good run.

IDENTIFY WHICH OBJECT THE ARM IS COMMITTING TO
The sub-task names ONE target object. Before you judge progress, name the object
the gripper is actually over or closing on RIGHT NOW — the one object directly
beneath or closest to the jaws — from what you see, NOT from what the sub-task
asked for. If the arm is reaching toward, descending onto, or grasping an object
that is NOT the named target, that is a wrong-object fault, even if everything
else looks like a clean approach. Do not excuse it because "something" is being
picked up; picking up the wrong thing is exactly the failure you catch.

WHAT COUNTS AS A FAULT (illegal transitions — make the concern concrete)
- transit -> approach / object no longer held: dropped or lost grip.
- repeated grasp -> approach with nothing acquired: failed grasps.
- place -> transit with the object still held: failed release.
- arm moving AWAY from the sub-task target for a sustained stretch: drift.
- committing to the WRONG object: the gripper is over / descending onto / closing
  on an object that is NOT the one the sub-task names. Judge this from
  object_under_gripper vs the named target, not from intent.
- knocking over / pushing away the target, or erratic motion that could damage.

NOT a fault: slow or indirect reaching, still starting up (little motion yet),
being mid-grasp or mid-transit, brief ambiguity you cannot resolve. Reaching over
a NEIGHBOURING object on the way to the target is only a fault once the gripper is
clearly descending onto or closing on the wrong object, not while merely passing
above it.

Scene context: the tray in the scene is the destination ("the tray" always means
it). The orange arm is part of the setup.

OUTPUT — OBSERVE FIRST, LABEL AFTER
Return ONLY one JSON object. The field order is deliberate: you must describe what
you literally see in the CURRENT FRAME *before* you name the phase, and the phase
must follow from what you described. Do NOT decide the phase first and then
back-fill the observations to match it.
{
  "gripper_visible": "<what the gripper is doing in the CURRENT FRAME: open, or closed on something>",
  "object_held": "<yes/no — is the target object clamped in the gripper jaws right now? Look at the current frame>",
  "object_under_gripper": "<name the ONE object directly beneath or closest to the jaws right now — what it actually is (e.g. plush toy, pouch, banana), or 'none' if the jaws are over bare table>",
  "target_match": "<yes/no — is object_under_gripper the SAME object the sub-task names?>",
  "object_status": "<where the target object is: on the table / in the gripper / on the tray>",
  "gripper_status": "<open | closed | transitioning>",
  "last_transition": "<the phase change since your prior look, and whether it was legal>",
  "phase": "<current stage, using the arc vocab — it MUST be consistent with the fields above>",
  "concern": "<none, or the specific expectation-violation>",
  "action": "CONTINUE" or "STOP",
  "reason": "<brief, concrete evidence from the current frame>"
}

CONSISTENCY RULES (the phase cannot contradict what you observed)
- object_held = yes  =>  you are AT LEAST at "grasp", and if the arm is moving the
  held object you are in "transit". You are NOT in "approach" — approach ends the
  moment the object is held.
- gripper closed on the object  =>  the grasp already happened; do not report
  "approach" or "nothing held".
- object_held = no AND gripper open AND arm reaching  =>  "approach".
- target_match = no AND the gripper is descending onto or closing on
  object_under_gripper  =>  wrong-object fault, action MUST be "STOP".

Examples:
{"gripper_visible": "closed on the pouch", "object_held": "yes", "object_under_gripper": "pouch", "target_match": "yes", "object_status": "pouch clamped in the gripper, lifted off the table", "gripper_status": "closed", "last_transition": "grasp -> transit (legal)", "phase": "transit", "concern": "none", "action": "CONTINUE", "reason": "Current frame shows the pouch held in the closed gripper and lifting; the grasp is done."}
{"gripper_visible": "open, descending toward the banana", "object_held": "no", "object_under_gripper": "banana", "target_match": "yes", "object_status": "banana on the table under the gripper", "gripper_status": "open", "last_transition": "approach -> approach (legal)", "phase": "approach", "concern": "none", "action": "CONTINUE", "reason": "Open gripper still lowering onto the banana; not yet grasped."}
{"gripper_visible": "open, descending onto the plush toy", "object_held": "no", "object_under_gripper": "plush toy", "target_match": "no", "object_status": "banana still on the table further back, not under the gripper", "gripper_status": "open", "last_transition": "approach onto the wrong object", "phase": "approach", "concern": "arm is descending onto the plush toy, but the sub-task names the banana", "action": "STOP", "reason": "The jaws are lowering onto the plush toy; the banana sits further back and is not the object being approached."}
{"gripper_visible": "open, empty", "object_held": "no", "object_under_gripper": "none", "target_match": "no", "object_status": "banana back on the table, short of the tray", "gripper_status": "open", "last_transition": "transit -> approach with object no longer held (illegal: dropped)", "phase": "approach", "concern": "grip lost in transit; banana fell short of the tray", "action": "STOP", "reason": "Prior look reported transit holding the banana, but the current frame shows an empty open gripper and the banana back on the table."}
"""

# Pass 1 of the observe->judge split: describe ONLY, no verdict. This removes the
# incentive to back-fill observations to match a verdict already chosen.
MONITOR_OBSERVE_SYSTEM_PROMPT = """You are the perception half of a real-time robot trajectory monitor. Your ONLY job on this pass is to describe what you literally see in the CURRENT FRAME. You do NOT decide CONTINUE or STOP — a separate judgement pass does that from your description. Do not mention verdicts.

WHAT YOU RECEIVE
- The sub-task the arm is attempting (context only — describe what you SEE, not what the sub-task implies should be there).
- The expected PHASE ARC for this kind of sub-task: the ordered stages a correct attempt passes through. Use its vocabulary to name the current phase.
- The frames in time order (earliest first, latest last). The latest frame is the current scene. Use the earlier frames only to read the direction of motion and the transition since the prior look.
- A PRIOR ASSESSMENT from the previous look. It is a hypothesis to CHECK against the frames, not ground truth — if the frames contradict it, believe the frames.

Ground every field in the latest (current) frame. Name only what you can actually see — object positions, whether the gripper holds anything. Report the phase that FOLLOWS from what you see; do not pick a phase first and describe to match it.

IDENTIFY WHICH OBJECT THE ARM IS COMMITTING TO
The sub-task names ONE target object. Name the object the gripper is actually over
or closing on RIGHT NOW — the one object directly beneath or closest to the jaws —
from what you SEE, NOT from what the sub-task asked for. Then say whether it is the
same object the sub-task names. This is a plain observation; the judgement pass
decides what to do with it.

CONSISTENCY RULES (the phase cannot contradict what you observed)
- object_held = yes  =>  you are AT LEAST at "grasp"; if the held object is moving you are in "transit", NOT "approach".
- gripper closed on the object  =>  the grasp already happened; do not report "approach" or "nothing held".
- object_held = no AND gripper open AND arm reaching  =>  "approach".

Return ONLY one JSON object, no verdict fields:
{
  "gripper_visible": "<what the gripper is doing in the current frame: open, or closed on something>",
  "object_held": "<yes/no — is the target object clamped in the gripper jaws right now?>",
  "object_under_gripper": "<name the ONE object directly beneath or closest to the jaws right now — what it actually is (e.g. plush toy, pouch, banana), or 'none' if over bare table>",
  "target_match": "<yes/no — is object_under_gripper the SAME object the sub-task names?>",
  "object_status": "<where the target object is: on the table / in the gripper / on the tray>",
  "gripper_status": "<open | closed | transitioning>",
  "last_transition": "<the phase change since the prior look, and whether it was legal>",
  "phase": "<current stage, using the arc vocab — consistent with the fields above>"
}
"""

# Pass 2 of the observe->judge split: verdict ONLY, from the pass-1 description
# plus the image. Kept deliberately narrow — the fault list and the "prefer to
# miss a marginal case" bias are here, where the decision is actually made.
MONITOR_JUDGE_SYSTEM_PROMPT = """You are the judgement half of a real-time robot trajectory monitor. A perception pass has already described the current frame (its observations are given to you). Your ONLY job is to decide whether the attempt is going wrong. You are NOT the success judge — never call the task complete. Verify the observations against the image; if the image contradicts them, believe the image.

You catch an attempt going wrong *before* it finishes on a bad premise.

VERDICT — binary
- "CONTINUE": the attempt is progressing legally through the phase arc, OR you are unsure. Default here.
- "STOP" (fault): a CLEAR expectation-violation — an illegal transition or sustained wrong-direction motion. STOP preempts the arm, so a spurious STOP aborts a good run. Raise it ONLY on a clear violation, never on momentary ambiguity or a single flickery frame. Prefer to miss a marginal case and let the end-of-task judge catch it over aborting a good run.

WHAT COUNTS AS A FAULT (illegal transitions — make the concern concrete)
- transit -> approach / object no longer held: dropped or lost grip.
- repeated grasp -> approach with nothing acquired: failed grasps.
- place -> transit with the object still held: failed release.
- arm moving AWAY from the sub-task target for a sustained stretch: drift.
- committing to the WRONG object: object_under_gripper is NOT the object the
  sub-task names (target_match = no) AND the gripper is descending onto or closing
  on it. Judge this from the observations, not from intent — picking up the wrong
  object is a fault even when the approach otherwise looks clean.
- knocking over / pushing away the target, or erratic motion that could damage.

NOT a fault: slow or indirect reaching, still starting up (little motion yet), being mid-grasp or mid-transit, brief ambiguity you cannot resolve. Merely passing ABOVE a neighbouring object on the way to the target is not a wrong-object fault; it becomes one once the gripper is clearly descending onto or closing on the wrong object.

Scene context: the tray in the scene is the destination ("the tray" always means it). The orange arm is part of the setup.

Return ONLY one JSON object:
{
  "concern": "<none, or the specific expectation-violation>",
  "action": "CONTINUE" or "STOP",
  "reason": "<brief, concrete evidence from the current frame>"
}
"""


# ---------------------------------------------------------------------------
# NARROWED ("semantic") variant.
#
# Scopes the monitor to the semantic/state faults a kinematic monitor cannot see:
# (1) is the arm acting on the RIGHT object, (2) did the grasp actually take /
# is it still held, (3) did the object land at the destination. Motion quality
# (speed, oscillation, stall, drift) is DELEGATED to the velocity-threshold
# monitor and explicitly out of scope here. Shorter and faster than the full
# prompt, and drops the "sustained wrong-direction motion" language that is the
# main source of spurious STOPs off sampled frames.
# ---------------------------------------------------------------------------

MONITOR_SEMANTIC_SYSTEM_PROMPT = """You are a real-time robot STATE verifier for an orange tabletop arm (SO-101, 6-DOF). While the arm executes a pick-and-place sub-task, a SEPARATE kinematic monitor watches motion quality — speed, oscillation, stalls, drift — from the arm's proprioception. That is NOT your job and you must never comment on it. YOUR job is the semantic state the kinematics is blind to: is the arm acting on the RIGHT object, and is its grasp/placement state actually what it appears to claim. You are NOT the success judge (a separate module fires at the end); never call the task complete.

WHAT YOU RECEIVE EACH LOOK
- The sub-task, which names ONE target object.
- The expected PHASE ARC (context for naming the phase).
- A short video clip of the last few seconds plus the CURRENT FRAME on its own in
  full detail. The current frame is the true present state — ground every field in
  it. Use the clip ONLY to tell whether the target is held NOW versus was held
  before (grasp/lost-grip), NEVER to assess how the arm is moving.
- A PRIOR ASSESSMENT from your previous look — a hypothesis to check against the
  frames, not ground truth. If the frames contradict it, believe the frames.

THE THREE CHECKS (all are STATE facts, none is motion):
1. RIGHT OBJECT — name the object directly beneath or closest to the jaws right
   now, from what you SEE, not from what the sub-task asked for. If the gripper is
   descending onto or closing on an object that is NOT the named target, that is a
   fault.
2. GRASP INTEGRITY — if the gripper is closed and lifted but the jaws are visibly
   empty (nothing acquired), OR the prior look reported the target held but the
   current frame shows empty jaws with the object back on a surface (lost grip),
   that is a fault.
3. PLACEMENT — if the object has been released somewhere other than its
   destination (dropped short of the tray), or the target has been knocked over or
   pushed away, that is a fault.

VERDICT — binary
- "CONTINUE": the arm is acting on the correct object and its grasp/placement state
  is consistent, OR you are unsure. Default here.
- "STOP" (fault): a CLEAR semantic violation from the three checks above. STOP
  preempts the arm, so raise it ONLY on a clear state fault, never on momentary
  ambiguity or a single flickery frame.

NOT YOUR CALL — do NOT STOP for any of these (the kinematic monitor owns them):
reaching speed, indirect or curved paths, oscillation, stalls, motion stopping,
"drifting" or "moving away". Merely passing ABOVE a neighbouring object on the way
to the target is fine — it becomes a fault only once the gripper is clearly
descending onto or closing on the wrong object.

OUTPUT — OBSERVE FIRST, LABEL AFTER
Return ONLY one JSON object. Describe what you literally see in the CURRENT FRAME
before you name the phase or the verdict; the phase and verdict must follow from
what you described, not the other way round.
{
  "gripper_visible": "<what the gripper is doing in the CURRENT FRAME: open, or closed on something>",
  "object_under_gripper": "<name the ONE object directly beneath or closest to the jaws right now — what it actually is (e.g. plush toy, pouch, banana), or 'none' if over bare table>",
  "target_match": "<yes/no — is object_under_gripper the SAME object the sub-task names?>",
  "object_held": "<yes/no — is the target object clamped in the jaws right now?>",
  "object_status": "<where the target object is: on the table / in the gripper / on the tray>",
  "gripper_status": "<open | closed | transitioning>",
  "phase": "<current stage, using the arc vocab — consistent with the fields above>",
  "concern": "<none, or the specific STATE fault (wrong object / failed grasp / lost grip / bad placement)>",
  "action": "CONTINUE" or "STOP",
  "reason": "<brief, concrete evidence from the current frame>"
}

CONSISTENCY RULES
- object_held = yes  =>  you are at least at "grasp"/"transit", not "approach".
- gripper closed on the object  =>  the grasp already happened; do not report "nothing held".
- target_match = no AND the gripper is descending onto or closing on object_under_gripper  =>  wrong-object fault, action MUST be "STOP".

Examples:
{"gripper_visible": "open, descending toward the banana", "object_under_gripper": "banana", "target_match": "yes", "object_held": "no", "object_status": "banana on the table under the gripper", "gripper_status": "open", "phase": "approach", "concern": "none", "action": "CONTINUE", "reason": "Jaws lowering onto the banana, the named target; not yet grasped."}
{"gripper_visible": "open, descending onto the plush toy", "object_under_gripper": "plush toy", "target_match": "no", "object_held": "no", "object_status": "banana still on the table further back, not under the gripper", "gripper_status": "open", "phase": "approach", "concern": "wrong object: gripper is descending onto the plush toy, but the sub-task names the banana", "action": "STOP", "reason": "The jaws are lowering onto the plush toy; the banana sits further back and is not the object being approached."}
{"gripper_visible": "closed, lifted, jaws empty", "object_under_gripper": "none", "target_match": "no", "object_held": "no", "object_status": "banana still on the table where it started", "gripper_status": "closed", "phase": "transit", "concern": "failed grasp: gripper closed and lifted but nothing is in the jaws", "action": "STOP", "reason": "Prior look reported the banana grasped, but the current frame shows the closed gripper lifted with empty jaws and the banana still on the table."}
"""

MONITOR_SEMANTIC_OBSERVE_SYSTEM_PROMPT = """You are the perception half of a real-time robot STATE verifier. Your ONLY job on this pass is to describe what you literally see in the CURRENT FRAME. You do NOT decide CONTINUE or STOP, and you do NOT comment on how the arm is moving (a separate kinematic monitor owns motion). Describe state only.

WHAT YOU RECEIVE
- The sub-task, which names ONE target object (context — describe what you SEE, not what the sub-task implies).
- The expected PHASE ARC (use its vocabulary to name the phase).
- The frames; the latest is the current scene. Use earlier frames ONLY to tell whether the target is held NOW versus was held before — never to assess motion.
- A PRIOR ASSESSMENT to CHECK against the frames, not to trust.

Ground every field in the current frame. Above all, name the object the gripper is actually over or closing on RIGHT NOW — the one object directly beneath or closest to the jaws — from what you see, and say whether it is the object the sub-task names.

Return ONLY one JSON object, no verdict fields:
{
  "gripper_visible": "<what the gripper is doing in the current frame: open, or closed on something>",
  "object_under_gripper": "<name the ONE object directly beneath or closest to the jaws right now (e.g. plush toy, pouch, banana), or 'none' if over bare table>",
  "target_match": "<yes/no — is object_under_gripper the SAME object the sub-task names?>",
  "object_held": "<yes/no — is the target object clamped in the jaws right now?>",
  "object_status": "<where the target object is: on the table / in the gripper / on the tray>",
  "gripper_status": "<open | closed | transitioning>",
  "phase": "<current stage, using the arc vocab — consistent with the fields above>"
}
"""

MONITOR_SEMANTIC_JUDGE_SYSTEM_PROMPT = """You are the judgement half of a real-time robot STATE verifier. A perception pass has already described the current frame (its observations are given to you). Your ONLY job is to decide whether a SEMANTIC state fault has occurred. You do NOT judge motion — speed, oscillation, stalls and drift belong to a separate kinematic monitor. You are NOT the success judge; never call the task complete. Verify the observations against the image; if the image contradicts them, believe the image.

Raise "STOP" only for a CLEAR fault from these three, otherwise "CONTINUE" (also when unsure):
1. WRONG OBJECT — target_match = no AND the gripper is descending onto or closing on object_under_gripper. Picking up the wrong object is a fault even when the approach otherwise looks clean.
2. FAILED GRASP / LOST GRIP — the gripper is closed and lifted but the jaws are empty, or the object that was reported held is now back on a surface with empty jaws.
3. BAD PLACEMENT — the object was released short of / off the destination, or the target was knocked over or pushed away.

Do NOT STOP for: reaching speed, indirect paths, oscillation, stalls, "moving away", or merely passing above a neighbouring object en route to the target. STOP preempts the arm — never fire on momentary ambiguity or a single flickery frame.

Scene context: the tray in the scene is the destination ("the tray" always means it). The orange arm is part of the setup.

Return ONLY one JSON object:
{
  "concern": "<none, or the specific STATE fault (wrong object / failed grasp / lost grip / bad placement)>",
  "action": "CONTINUE" or "STOP",
  "reason": "<brief, concrete evidence from the current frame>"
}
"""


# ---------------------------------------------------------------------------
# Phase arcs (copied from vlm_robot_orchestrator.py so this harness matches what
# the monitor is actually told, without importing the orchestrator's heavy
# robot/lerobot dependency stack).
# ---------------------------------------------------------------------------

_PHASE_ARCS = {
    "pick_and_place": (
        "approach -> grasp -> transit -> place -> retreat",
        "The arm approaches the target, grasps it (gripper closes on it), carries "
        "it in transit toward the destination, places it (gripper opens to release "
        "over the destination), then retreats. Legal progress moves forward through "
        "these stages; jumping backward (e.g. transit -> approach with the object no "
        "longer held) is a fault.",
    ),
    "wipe": (
        "in_progress -> covered_region_grew -> (repeat) ",
        "The arm sweeps across a region; there is no grasp/place/retreat. Legal "
        "progress is the covered region growing over time. A sustained stall "
        "(motion stopped with the region not yet covered) is the fault to watch.",
    ),
}


def monitor_phase_arc(sub_task: str) -> tuple[str, str]:
    """Pick the expected phase arc (vocab, description) for a sub-task's primitive.

    'wipe'/'clean'/'sweep' select the wipe arc; everything else defaults to
    pick-and-place. Mirrors vlm_robot_orchestrator.monitor_phase_arc.
    """
    t = sub_task.lower()
    if any(k in t for k in ("wipe", "clean", "sweep")):
        return _PHASE_ARCS["wipe"]
    return _PHASE_ARCS["pick_and_place"]


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
# User-message assembly
# ---------------------------------------------------------------------------

# Explains the single-still contact-sheet input so the copied prompt's references
# to "a video clip" and "the CURRENT FRAME on its own" stay coherent.
GRID_IMAGE_NOTE = (
    "The image is a CONTACT SHEET of the last few seconds: several frames tiled in "
    "time order and read left-to-right, top-to-bottom, each labelled with its "
    "timestamp (t=..s #i). The EARLIEST frame is #0; the LAST tile (highest #) is "
    "the CURRENT FRAME — the scene right now. Read the direction of motion across "
    "the tiles, then ground your description in that last tile."
)


def _prior_block(prior_state: dict | None) -> str:
    """Render the PRIOR ASSESSMENT block, matching the orchestrator's wording."""
    if prior_state:
        return (
            "PRIOR ASSESSMENT (your previous look, the moment just before the first "
            "tile). This is very likely STALE — the arm has moved since. It is only "
            "context for the transition; the CURRENT FRAME overrides it in every "
            "case of disagreement:\n"
            f"  phase: {prior_state.get('phase', '?')}\n"
            f"  object_status: {prior_state.get('object_status', '?')}\n"
            f"  gripper_status: {prior_state.get('gripper_status', '?')}\n"
            f"  concern: {prior_state.get('concern', 'none')}\n"
        )
    return (
        "PRIOR ASSESSMENT: none — this is the FIRST look at this sub-task. The arm "
        "may still be starting up; report the phase you observe and default to "
        "CONTINUE unless there is already a clear fault.\n"
    )


def _monitor_user_content(image: Image.Image, sub_task: str,
                          prior_state: dict | None,
                          closing: str) -> list[dict]:
    """Build the shared user content: grid note + image + task/arc/prior + closing.

    `closing` is the pass-specific final instruction (describe vs judge vs both).
    """
    arc_vocab, arc_desc = monitor_phase_arc(sub_task)
    text = (
        f'\nThe robot is currently attempting this sub-task: "{sub_task}"\n\n'
        f"EXPECTED PHASE ARC for this sub-task:\n  {arc_vocab}\n  {arc_desc}\n\n"
        f"{_prior_block(prior_state)}\n"
        f"{GRID_IMAGE_NOTE}\n\n"
        f"{closing}"
    )
    return [
        {"type": "text", "text": "[top-down camera — monitor contact sheet]"},
        {"type": "image", "image": image},
        {"type": "text", "text": text},
    ]


# ---------------------------------------------------------------------------
# The monitor call(s)
# ---------------------------------------------------------------------------

def run_monitor_single(model, processor, prompts, image, sub_task,
                       prior_state, temperature):
    """Single-call monitor: observations + verdict in one shot.

    Returns (parsed_dict_or_None, raw_output).
    """
    content = _monitor_user_content(
        image, sub_task, prior_state,
        closing=("Fill every field of the JSON. Is the attempt progressing legally "
                 "through the arc (CONTINUE), or is there a clear "
                 "expectation-violation (STOP)?"),
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompts["single"]}]},
        {"role": "user", "content": content},
    ]
    output = generate(model, processor, messages,
                      max_new_tokens=MONITOR_MAX_NEW_TOKENS, temperature=temperature)
    return parse_json_object(output), output


def run_monitor_split(model, processor, prompts, image, sub_task,
                      prior_state, temperature):
    """observe->judge monitor: pass 1 describes, pass 2 judges the description.

    Returns (merged_dict_or_None, raw_output_string). The merged dict combines the
    observation fields (pass 1) and the verdict fields (pass 2) so the printout
    matches the single-call schema.
    """
    # --- Pass 1: observe ---
    obs_content = _monitor_user_content(
        image, sub_task, prior_state,
        closing="Describe the current frame. Fill every field of the observation "
                "JSON. Do NOT give a verdict.",
    )
    obs_messages = [
        {"role": "system", "content": [{"type": "text", "text": prompts["observe"]}]},
        {"role": "user", "content": obs_content},
    ]
    print("  [pass 1 — observe]")
    obs_output = generate(model, processor, obs_messages,
                          max_new_tokens=MONITOR_MAX_NEW_TOKENS, temperature=temperature)
    observations = parse_json_object(obs_output) or {}

    # --- Pass 2: judge, given the observations ---
    obs_summary = json.dumps(observations, indent=2) if observations else obs_output.strip()
    judge_content = _monitor_user_content(
        image, sub_task, prior_state,
        closing=("The perception pass reported these observations of the current "
                 f"frame:\n{obs_summary}\n\nVerify them against the image, then "
                 "return the verdict JSON (concern, action, reason)."),
    )
    judge_messages = [
        {"role": "system", "content": [{"type": "text", "text": prompts["judge"]}]},
        {"role": "user", "content": judge_content},
    ]
    print("  [pass 2 — judge]")
    judge_output = generate(model, processor, judge_messages,
                            max_new_tokens=MONITOR_MAX_NEW_TOKENS, temperature=temperature)
    verdict = parse_json_object(judge_output) or {}

    merged = dict(observations)
    merged.update(verdict)
    raw = (f"--- pass 1 (observe) ---\n{obs_output}\n\n"
           f"--- pass 2 (judge) ---\n{judge_output}")
    return (merged if merged else None), raw


def run_monitor(model, processor, prompts, image, sub_task, prior_state,
                temperature, split):
    """Dispatch to the single-call or observe->judge path and pretty-print it.

    Returns the merged/parsed verdict dict (or {} on parse failure) so the REPL
    can thread it forward as the next call's prior_state.
    """
    print(f"\n{'='*60}")
    print(f"MONITOR CHECK ({'observe->judge' if split else 'single-call'})")
    print(f"Sub-task: \"{sub_task}\"")
    # (variant is reflected in the loaded prompts, shown via /prompt and /help)
    arc_vocab, _ = monitor_phase_arc(sub_task)
    print(f"Phase arc: {arc_vocab}")
    print(f"Prior: {prior_state if prior_state else 'none (first look)'}")
    print(f"{'='*60}")

    runner = run_monitor_split if split else run_monitor_single
    parsed, raw = runner(model, processor, prompts, image, sub_task,
                         prior_state, temperature)

    print(f"\nModel output:\n{raw}")

    if parsed is None:
        print("\n[WARNING] Could not parse a JSON verdict — production would "
              "fail-safe to CONTINUE here.")
        return {}

    action = str(parsed.get("action", "?")).upper()
    print(f"\n{'-'*60}")
    print(f"  VERDICT: {action}")
    print(f"  phase:          {parsed.get('phase', '?')}")
    print(f"  object_held:    {parsed.get('object_held', '?')}")
    print(f"  object_status:  {parsed.get('object_status', '?')}")
    print(f"  gripper_status: {parsed.get('gripper_status', '?')}")
    print(f"  last_transition:{parsed.get('last_transition', '?')}")
    print(f"  concern:        {parsed.get('concern', '?')}")
    print(f"  reason:         {parsed.get('reason', '?')}")
    print(f"{'-'*60}")
    return parsed


# ---------------------------------------------------------------------------
# Prompt file / editor helpers
# ---------------------------------------------------------------------------

# A --prompt-file bundles all three prompts in one text file, split by these
# markers, so /reload can refresh them without touching the model.
_PROMPT_MARKERS = {
    "single": "### SINGLE ###",
    "observe": "### OBSERVE ###",
    "judge": "### JUDGE ###",
}


# Two scopings you can A/B with /variant (or --variant at startup):
#   full     — the original monitor: right object + grasp/placement state AND
#              motion faults (drift, wrong-direction) all in the VLM.
#   semantic — narrowed to the state faults a kinematic monitor cannot see
#              (right object, grasp integrity, placement); motion is delegated
#              to the velocity-threshold monitor.
PROMPT_VARIANTS = {
    "full": {
        "single": MONITOR_SYSTEM_PROMPT,
        "observe": MONITOR_OBSERVE_SYSTEM_PROMPT,
        "judge": MONITOR_JUDGE_SYSTEM_PROMPT,
    },
    "semantic": {
        "single": MONITOR_SEMANTIC_SYSTEM_PROMPT,
        "observe": MONITOR_SEMANTIC_OBSERVE_SYSTEM_PROMPT,
        "judge": MONITOR_SEMANTIC_JUDGE_SYSTEM_PROMPT,
    },
}


def default_prompts(variant: str = "full") -> dict:
    return dict(PROMPT_VARIANTS[variant])


def serialize_prompts(prompts: dict) -> str:
    return "\n".join(
        f"{_PROMPT_MARKERS[key]}\n{prompts[key].strip()}\n"
        for key in ("single", "observe", "judge")
    )


def parse_prompt_file(text: str, base: dict) -> dict:
    """Parse a prompt file into {single, observe, judge}.

    Sections are delimited by the markers above. Any section missing from the
    file keeps its value from `base`, so a file may override just one prompt.
    """
    result = dict(base)
    # Find each marker's position, then slice between them.
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
# Prior-state entry
# ---------------------------------------------------------------------------

def prompt_for_prior() -> dict | None:
    """Interactively collect a prior_state to thread into the next check.

    Accepts either a pasted JSON object or blank fields. Returns None (cleared)
    if the user enters nothing.
    """
    print("Paste a prior-state JSON object, or press Enter to fill fields "
          "individually (blank line to skip a field):")
    line = input("  json> ").strip()
    if line:
        parsed = parse_json_object(line)
        if parsed is None:
            print("  [WARNING] Not valid JSON — prior left unchanged.")
            return "unchanged"
        return parsed
    state = {}
    for field in ("phase", "object_status", "gripper_status", "concern"):
        val = input(f"  {field}> ").strip()
        if val:
            state[field] = val
    return state or None


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

INTERACTIVE_HELP = """
Commands:
  <any text>          Run the monitor on the current image for this sub-task
                      (e.g. "put the banana on the tray")
  /image <path>       Load a different saved frame/grid (current: {image})
  /variant <name>     Swap prompt scope: full | semantic. Resets any /edit
                      changes to that variant's built-in prompts. (current: {variant})
  /split              Toggle single-call vs observe->judge (current: {split})
  /temp <value>       Set temperature (current: {temp})
  /prior              Set/clear the PRIOR ASSESSMENT threaded into the next check
                      (current: {prior})
  /auto               Toggle auto-threading: feed each verdict's state as the next
                      check's prior, as the orchestrator does (current: {auto})
  /prompt [which]     Print a prompt: single | observe | judge | all (default all)
  /edit [which]       Edit a prompt live in $EDITOR: single | observe | judge
  /reload             Reload prompts from --prompt-file (current: {pfile})
  /save <path>        Write the current prompts to a file (reloadable with /reload)
  /help               Show this help
  /quit               Exit
""".strip()


def interactive_loop(model, processor, prompts, image, image_path,
                     prompt_file, temp, split, variant="full"):
    prior_state = None
    auto_thread = False

    print(f"\n{'='*60}")
    print("MONITOR TUNING MODE — model loaded, type a sub-task to run a check.")
    print(f"Image: {image_path or '(none — set with /image)'}")
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
                image=image_path or "none",
                variant=variant,
                split="observe->judge" if split else "single-call",
                temp=temp,
                prior=(json.dumps(prior_state) if prior_state else "none"),
                auto="ON" if auto_thread else "OFF",
                pfile=prompt_file or "none",
            ))

        elif user_input.startswith("/variant"):
            parts = user_input.split(maxsplit=1)
            which = parts[1].strip() if len(parts) > 1 else ""
            if which not in PROMPT_VARIANTS:
                print(f"Usage: /variant {' | '.join(PROMPT_VARIANTS)}  "
                      f"(current: {variant})")
                continue
            variant = which
            prompts = dict(PROMPT_VARIANTS[which])
            print(f"Prompt variant: {variant} (reset to built-in prompts; any "
                  "/edit changes discarded).")

        elif user_input.startswith("/image"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("Usage: /image <path>")
                continue
            p = Path(parts[1].strip())
            if not p.exists():
                print(f"[ERROR] Image not found: {p}")
                continue
            try:
                image = Image.open(p).convert("RGB")
                image_path = str(p)
                print(f"Loaded image: {p} ({image.width}x{image.height})")
            except Exception as e:  # noqa: BLE001
                print(f"[ERROR] Could not open image: {e}")

        elif user_input == "/split":
            split = not split
            print(f"Pass structure: {'observe->judge' if split else 'single-call'}")

        elif user_input.startswith("/temp "):
            try:
                temp = float(user_input[6:].strip())
                print(f"Temperature set to {temp}")
            except ValueError:
                print("Usage: /temp <float>  (e.g. /temp 0.1)")

        elif user_input == "/prior":
            result = prompt_for_prior()
            if result == "unchanged":
                pass
            elif result is None:
                prior_state = None
                print("Prior cleared (next check is a first look).")
            else:
                prior_state = result
                print(f"Prior set: {json.dumps(prior_state)}")

        elif user_input == "/auto":
            auto_thread = not auto_thread
            print(f"Auto-threading: {'ON' if auto_thread else 'OFF'}")

        elif user_input.startswith("/prompt"):
            parts = user_input.split(maxsplit=1)
            which = parts[1].strip() if len(parts) > 1 else "all"
            keys = ("single", "observe", "judge") if which == "all" else (which,)
            for key in keys:
                if key not in prompts:
                    print(f"Unknown prompt: {key}. Use single | observe | judge | all")
                    continue
                print(f"\n{'#'*60}\n# {key.upper()} PROMPT\n{'#'*60}\n{prompts[key]}")

        elif user_input.startswith("/edit"):
            parts = user_input.split(maxsplit=1)
            which = parts[1].strip() if len(parts) > 1 else ""
            if which not in prompts:
                print("Usage: /edit single | observe | judge")
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

        # --- Run a monitor check ---
        else:
            if image is None:
                print("[ERROR] No image loaded. Use /image <path> or start with "
                      "--image.")
                continue
            verdict = run_monitor(model, processor, prompts, image, user_input,
                                  prior_state, temp, split)
            if auto_thread and verdict:
                prior_state = {k: verdict.get(k) for k in
                               ("phase", "object_status", "gripper_status",
                                "last_transition", "concern")
                               if verdict.get(k) is not None}
                print(f"[auto] prior for next check: {json.dumps(prior_state)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive harness for tuning the Qwen3-VL continuous monitor"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-VL-4B-Instruct",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--image", default=None,
        help="Path to a saved frame/monitor grid (e.g. runs/<run>/<task>/monitor_*.png)",
    )
    parser.add_argument(
        "--variant", default="full", choices=list(PROMPT_VARIANTS),
        help="Prompt scope: 'full' (state + motion) or 'semantic' (state only, "
             "motion delegated to the kinematic monitor). Default: full",
    )
    parser.add_argument(
        "--split", action="store_true",
        help="Start in observe->judge mode (default: single-call)",
    )
    parser.add_argument(
        "--temp", type=float, default=0.1,
        help="Sampling temperature (default: 0.1)",
    )
    parser.add_argument(
        "--prompt-file", default=None,
        help="Load prompt overrides from this file (editable, reloadable with /reload)",
    )
    args = parser.parse_args()

    # --- Resolve prompts ---
    prompts = default_prompts(args.variant)
    prompt_file = args.prompt_file
    if prompt_file:
        p = Path(prompt_file)
        if p.exists():
            prompts = parse_prompt_file(p.read_text(), prompts)
            print(f"Loaded prompt overrides from {prompt_file}.")
        else:
            print(f"[INFO] --prompt-file {prompt_file} does not exist yet; using "
                  "built-in prompts. /save will create it.")

    # --- Resolve image ---
    image = None
    image_path = None
    if args.image:
        p = Path(args.image)
        if not p.exists():
            print(f"[ERROR] Image not found: {p}")
            return
        image = Image.open(p).convert("RGB")
        image_path = str(p)
        print(f"Using image: {p} ({image.width}x{image.height})")
    else:
        print("[INFO] No --image given; set one with /image before running a check.")

    # --- Load model ---
    model, processor = load_model(args.model)

    interactive_loop(model, processor, prompts, image, image_path,
                     prompt_file, args.temp, args.split, variant=args.variant)


if __name__ == "__main__":
    main()
