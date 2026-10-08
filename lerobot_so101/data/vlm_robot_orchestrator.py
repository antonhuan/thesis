# vlm_robot_orchestrator.py
#
# Dual-system orchestrator: connects the VLM reasoning layer (vlm.py) to the
# VLA execution layer (robot_client_loop.py).
#
#   high-level prompt
#        |
#        v
#   [VLM: Qwen3-VL]  -- decompose -->  queue of sub-tasks
#        |
#        v
#   [VLA: policy server + LoopRobotClient]  -- one episode per sub-task
#        |      [VLM monitor: at the grasp, is the arm going for the RIGHT object?
#        |       (vlm_monitor_live.LiveMonitor) -- STOP ends the episode]
#        v
#   [VLM: success evaluation]  -- retry on failure, replan if retries exhausted
#        |
#        v
#   [VLM: final whole-task evaluation]  -- if it judges the high-level prompt
#        incomplete, plan again over the objects its scene survey found not yet
#        on the tray and run the whole loop over (up to --max_final_retries rounds)
#
# Usage:
#   Terminal 1 (policy server, stays running):
#     python fast_policy_server.py --host=127.0.0.1 --port=8080
#     (same flags as `python -m lerobot.async_inference.policy_server`, but skips
#      the ~3 min random init pi05 otherwise pays for on every load)
#
#   Terminal 2 (this script):
#     python vlm_robot_orchestrator.py \
#       --robot.type=so101_follower \
#       --robot.port=/dev/ttyACM0 \
#       --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: intelrealsense, serial_number_or_name: 342222071104, width: 640, height: 480, fps: 30}}" \
#       --task="idle" \
#       --server_address=127.0.0.1:8080 \
#       --policy_type=smolvla \
#       --pretrained_name_or_path=edge-inference/smolvla-so101-pick-orange \
#       --policy_device=cuda \
#       --actions_per_chunk=50 \
#       --chunk_size_threshold=0.7 \
#       --episode_duration=100 \
#       --vlm_model=Qwen/Qwen3-VL-8B-Instruct \
#       --evaluate_subtasks=true \
#       --vlm_eval_use_video=true \
#       --vlm_eval_num_frames=16 \
#       --clip_buffer_maxlen=128 \
#       --max_retries=3 \
#       --max_replans=1 \
#       --max_final_retries=1 \
#       --enable_interjection=true
#
# At the prompt, type a HIGH-LEVEL instruction (e.g. "clean up the table but
# leave the cups"). The VLM decomposes it into sub-tasks, each sub-task is
# executed by the VLA policy, and (optionally) the VLM judges success from a
# fresh camera frame, retrying failed sub-tasks. If retries are exhausted the
# VLM replans the remaining sequence given the current scene state.
#
# Once the queue empties, the VLM judges the ORIGINAL instruction against the
# final scene. If that verdict is a failure, the prompt is planned again — over
# only the objects the final evaluation's scene survey found not yet on the tray,
# with the evaluator's reason and the verified-completed sub-tasks as context
# (--targeted_retry_decompose; stops if the survey finds nothing left) — and the
# whole queue/execute/evaluate loop runs again, for up to --max_final_retries
# extra rounds. All rounds of one prompt share a single task folder and log.
#
# Every replan pauses and asks you for additional context to pass to the VLM
# (e.g. "the toy is behind the cup"). Press Enter alone to replan without it.
# Note this blocks: an unattended run waits at a replan until someone responds.
#
# Press 'q'+Enter at any point during execution (or at a replan prompt) to ABORT
# the whole high-level task: the current episode stops, the arm returns home, and
# you are asked for a new high-level prompt. This works regardless of
# --enable_interjection.
#
# Type 'quit' or 'exit' to shut down cleanly.

import enum
import json
import logging
import re
import select
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

import numpy as np
from PIL import Image

from lerobot.utils.import_utils import register_third_party_plugins

import draccus

from robot_client_loop import LoopClientConfig, LoopRobotClient
from vlm_core import generate, load_model
from vlm_monitor_live import LiveMonitor, MonitorSettings, TappedClient
from vlm import (
    DECOMPOSITION_SYSTEM_PROMPT,
    REFINEMENT_SYSTEM_PROMPT,
    identify_objects,
    survey_scene,
    survey_scene_change,
    decompose_from_objects,
)
from datetime import datetime

# ---------------------------------------------------------------------------
# Config: extends the loop client config with VLM parameters
# ---------------------------------------------------------------------------
@dataclass
class OrchestratorConfig(MonitorSettings, LoopClientConfig):
    # Keep one actions.log per high-level task (written beside task.log by
    # run_task via start_action_log), which parse_episodes.py splits by episode
    # markers -- rather than one per VLA episode directory.
    action_log_per_episode: bool = True
    # HuggingFace model name or local path for the VLM planner
    vlm_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    # Sampling temperature for VLM generation
    vlm_temperature: float = 0.1
    # Split decomposition into two model calls against the same frame:
    # (1) identify visible objects, then (2) decompose given that object list.
    # Set false to use the single-call DECOMPOSITION_SYSTEM_PROMPT instead.
    vlm_two_pass_decompose: bool = True
    # Re-ask the planner once when it lists an object in allowed_objects but gives
    # it no sub-task. Off by default: on first-round plans it never recovered a
    # dropped object and turned every count-limited instruction ("put one object
    # on the tray") into moving everything (see vlm_decompose_test.py).
    vlm_reask_uncovered: bool = False
    # Two-pass decompose: the identified objects reach the planner the way a targeted
    # retry round's do — fixtures (SURVEY_FIXTURE_WORDS: the arm, its wires and label,
    # ...) removed, the tray and table kept as destinations — and any sub-task that would
    # move a fixture is dropped. The identify prompt already says the arm is not an
    # object; on the decompose test it still drew 27 "put the robot arm/wire on the
    # tray" sub-tasks in 195 cases.
    vlm_filter_fixtures: bool = True
    # Use the dedicated top-down RealSense D435 for VLM input.
    # If false (or the camera fails to start), falls back to the robot's own
    # camera observation, then to text-only.
    vlm_camera_key: str = "front"
    # After each sub-task episode, ask the VLM whether it succeeded
    evaluate_subtasks: bool = True
    # After the whole sub-task queue finishes, ask the VLM whether the ORIGINAL
    # high-level instruction was accomplished, judging the final scene against the
    # prompt. Runs once per round; skipped on abort.
    evaluate_final: bool = True
    # How many extra rounds to run when the final evaluation judges the whole
    # high-level task a failure: the prompt is decomposed again against the
    # current scene (with the final evaluator's reason as context) and the entire
    # queue -> execute -> evaluate loop runs again. 0 disables the behaviour
    # (final eval stays log-only). Requires evaluate_final.
    max_final_retries: int = 1
    # Judge the final scene against the scene before the robot started (the round-1
    # decompose frame): a BEFORE/AFTER survey names every object from the start state
    # (so stacked and hidden objects keep their row), stacks are resolved in code, a
    # call that sees only the instruction, the object names and the BEFORE frame —
    # never the final locations — decides which objects must / must not end on the
    # tray, and code compares. False keeps the single-image survey -> judge path.
    # Measured on the 200-case vlm_final_eval_test sample: 0.71 -> 0.90 correct.
    final_eval_before_after: bool = True
    # When the arm did not move (no_movement), check before the sub-task evaluator
    # whether the sub-task's object already rests on the tray: the same BEFORE/AFTER
    # survey (the clip's first and last frames, no instruction), stacks resolved in
    # code, the requirements call to pick out the object the sub-task means, code
    # decides. If it is on the tray the sub-task counts as done and the evaluator is
    # skipped — the evaluator fails such an attempt as "the arm did not move". Only on
    # no-movement attempts: run on every attempt, survey misreads (an object beside
    # the tray, or held over it) became false passes (vlm_subtask_eval_test: 13 vs 5).
    subtask_done_precheck: bool = True
    # On a retry round, plan from the final evaluation's scene survey instead of a
    # fresh identify pass: the planner is handed only the objects the survey found
    # NOT yet at a destination (plus the destinations), so it re-plans the work
    # that remains and cannot invent sub-tasks for the robot arm, table or tray.
    # If the survey finds nothing off a destination, the task stops. Falls back to
    # a full re-decompose when there is no survey (single-call final eval).
    targeted_retry_decompose: bool = True
    # Show the evaluator a short video clip of the episode (buffered during
    # execution) instead of a single still frame. Falls back to a still if the
    # clip buffer is empty (e.g. a very short / no-movement episode).
    vlm_eval_use_video: bool = True
    # Number of frames sampled (evenly in time, across the whole episode) from the
    # episode buffer for the eval clip.
    vlm_eval_num_frames: int = 16
    # Nominal frames-per-second passed to the VLM for the eval clip (the buffer
    # is captured at the control loop's observation rate, not a fixed fps).
    vlm_eval_fps: float = 2.0
    # How many times to re-run a sub-task the VLM judged as failed
    max_retries: int = 3
    # How many times to replan the remaining sequence after a sub-task fails
    # all retries. Set to 0 to disable replanning.
    max_replans: int = 1
    # Save every frame sent to the VLM under ./frames/
    save_frames: bool = False
    # Enable the background stdin listener that lets the user skip/replan
    # mid-execution (Enter/'s' to skip, 'r'+Enter to replan). Off by default;
    # pass --enable_interjection=true to enable. 'q'+Enter aborts either way.
    enable_interjection: bool = False
    # --- Vocabulary refinement (closed-loop failure classification) ---
    enable_vocab_refinement: bool = True
    # --- VLM monitor during VLA execution (vlm_monitor_live.LiveMonitor) ---
    # Is the arm going for the RIGHT object? One check per episode at the grasp: the
    # client freezes the arm the instant the gripper closes on an object, a two-pass
    # identify -> compare check judges the look-back window over the approach (front +
    # wrist), and a STOP — or an unclear reach — is confirmed or vetoed by a wrist probe
    # against the sub-task's object grounded on the episode's first frame. A STOP ends
    # the episode as a failure (no evaluator call): the same instruction is retried
    # once, and a second STOP skips the sub-task (no replan). Needs a wrist camera
    # (`wrist_key`). Its settings (gate, window, verify, match, grasp freeze, release
    # ...) are inherited from vlm_monitor_live.MonitorSettings, with the defaults
    # measured on the robot (50 live trials: 92% specificity, 100% recall).
    enable_vlm_monitor: bool = True
    # Episode timeout. It counts wall time, including the monitor's freezes (the reach
    # check and the wrist verify hold the arm for ~15-30 s each), so it has to be well
    # above the VLA's own run time; the live monitor trials used 100 s.
    episode_duration: float = 100.0


# ---------------------------------------------------------------------------
# System prompt for success evaluation
# ---------------------------------------------------------------------------
EVALUATION_SYSTEM_PROMPT = """You are a robot task evaluator. You receive a short video clip of a robot arm attempting a sub-task, and the sub-task text. The clip's frames are in time order: earlier frames show the start of the attempt, later frames show the end.

Assess whether the sub-task was completed successfully. Judge the FINAL state (the last frames), but use the motion across the clip as evidence — e.g. whether the object was actually grasped, moved, and released at the destination rather than dropped or knocked aside.

If the task was a failure, use the video clip to include details about how the failure occurred and include it in the reason.

Scene context:
- The tray visible in the scene is the destination. It may be any colour (pink, black, etc.). "the tray" in the sub-task always means this tray.
- "away" means onto the tray.
- The orange robot arm is part of the setup, ignore it.
- Judge ONLY whether the specific object named in the sub-task is now at the destination. Do not assess other objects. If the object is partially in the destination, count it as a success.
- The arm begins from a resting position. Focus on the directed motion phase when classifying failure — do not interpret the initial stationary period as hesitation.

Output format:
Return ONLY a JSON object:
- "success": true or false
- "failure_type": one of "VOCAB" or "RETRY" (only when success is false)
- "reason": a brief explanation of your judgement

Failure classification (only when success is false):
- VOCAB: The arm moved aggressively or erratically toward the wrong object, OR moved with no clear target commitment with oscillatory movement (wandering, hesitation, small uncertain motions). This suggests the instruction label confused the robot.
- RETRY: The arm reached toward the correct object but the grasp or placement failed (e.g. gripper closed on air, object slipped, placed in wrong spot, dropped during transport). The instruction was understood but execution failed.

Example:
{"success": true, "reason": "The apple is now on the tray as instructed."}
{"success": false, "failure_type": "RETRY", "reason": "The arm reached the banana but the gripper closed on air and did not grasp it."}
{"success": false, "failure_type": "VOCAB", "reason": "The arm moved aggressively toward the pouch instead of the banana."}
"""


# ---------------------------------------------------------------------------
# System prompt for the final whole-task evaluation
# ---------------------------------------------------------------------------
FINAL_EVALUATION_SYSTEM_PROMPT = """You are a robot task evaluator. You receive a single camera image of a tabletop workspace after a robot arm has finished attempting a high-level instruction, plus the original instruction text. Judge whether the WHOLE instruction was accomplished, based only on the final scene.

Scene context:
- The tray visible in the scene is the destination. It may be any colour (pink, black, etc.). "the tray" and "away" in the instruction always mean this tray.
- The orange robot arm is part of the setup, ignore it. It is not an object.

Judge overall success as ALL of the following:
- Every object the instruction asked to move (or clear/put away/tidy) is now at its destination (e.g. on the tray).
- Every object the instruction asked to LEAVE, skip, keep, ignore, or not touch is still in its original place on the table and is NOT on the tray.
- Any arrangement or spatial relationship the instruction called for (e.g. "next to", "on top of") holds in the final scene.

If any of these is not met, the task failed.

Output format:
Return ONLY a JSON object:

{"reason": "<one or two sentences>", "success": true or false}

Rules:
- "reason": if success is false, name the specific object(s) that are not where the instruction requires, and where they actually are. State a conclusion only — do NOT deliberate, reconsider, re-read the instruction, or revise yourself inside this field. Never write "wait", "actually", or "I made a mistake".
- "success": true only if every object in the scene satisfies the instruction.

Examples:
{"reason": "The apple is on the tray and the banana was correctly left on the table.", "success": true}
{"reason": "The pouch is still on the table instead of on the tray.", "success": false}
"""


# ---------------------------------------------------------------------------
# System prompt for the judging half of the two-pass final evaluation
# ---------------------------------------------------------------------------
# Pass 2 of locate -> judge. The locations are handed in as findings from a
# separate perception pass that never saw the instruction, so this call does no
# looking at all: it only decides what the instruction REQUIRES and compares.
# Splitting it this way is the fix for the evaluator filling "location" in from
# "required" — it no longer gets to write "location".
FINAL_JUDGE_SYSTEM_PROMPT = """You are a robot task evaluator. A robot arm has finished attempting a high-level instruction on a tabletop workspace. You receive the original instruction, plus the findings of a separate perception module that inspected the final scene and reported where each object now rests.

The perception findings are GROUND TRUTH about where the objects are. You did not see the scene and cannot revise the findings — judge only against what they report.

Scene context:
- The tray is the destination. It may be any colour (pink, black, etc.). "the tray" and "away" in the instruction always mean this tray.
- The orange robot arm is part of the setup, ignore it. It is not an object.
- Be mindful that the object can appear to be on top of another object or leaning on it but still be inside the tray.
- The tray and robot arm are not considered objects that should be manipulated, do not consider them in the evaluation.

Judge overall success as ALL of the following:
- Every object the instruction asked to move (or clear/put away/tidy) is now at its destination (e.g. on the tray).
- Every object the instruction asked to LEAVE, skip, keep, ignore, or not touch is still on the table and is NOT on the tray.
- Any arrangement or spatial relationship the instruction called for (e.g. "next to", "on top of") holds in the findings.

If any of these is not met, the task failed.

Output format:
Return ONLY a JSON object:

{"reason": "<one or two sentences>", "success": true or false}

Rules:
- "reason": if success is false, name the specific object(s) from the findings that are not where the instruction requires, and where the findings say they actually are. State a conclusion only — do NOT deliberate, reconsider, re-read the instruction, or revise yourself inside this field. Never write "wait", "actually", or "I made a mistake".
- "success": true only if every object in the findings satisfies the instruction.

Examples:

Instruction: "put everything on the tray except for the banana"
Findings: [{"object": "apple", "support": "tray"}, {"object": "banana", "support": "table"}]
{"reason": "The apple is on the tray and the banana was correctly left on the table.", "success": true}

Instruction: "put everything on the tray except for the banana"
Findings: [{"object": "pouch", "support": "table"}, {"object": "plush toy", "support": "tray"}, {"object": "banana", "support": "table"}]
{"reason": "The pouch is still on the table instead of on the tray.", "success": false}
"""


# ---------------------------------------------------------------------------
# System prompt for replanning
# ---------------------------------------------------------------------------
REPLAN_SYSTEM_PROMPT = """You are a task planner for an orange tabletop robot arm (SO-101, 6-DOF). A previous plan partially failed and you need to produce a revised plan for the REMAINING work.

You will receive:
- The original user instruction (including any preferences)
- Which sub-tasks have already been completed successfully
- Which sub-task failed and why
- A current camera observation of the workspace

Produce a revised sub-task list that:
1. Does NOT repeat any already-completed sub-tasks.
2. Accounts for the current scene state visible in the camera.
3. Respects all preferences and constraints from the original instruction.
4. May rephrase or reorder the failed sub-task if a different approach could help.

You MUST respond in the following JSON format exactly:

{
  "visible_objects": ["list", "of", "objects", "you", "see"],
  "excluded_objects": ["objects", "the", "instruction", "says", "to", "leave"],
  "allowed_objects": ["visible", "minus", "excluded"],
  "subtasks": ["put the X on the tray", "put the Y on the tray"]
}

Rules:
- The orange robot arm is not an object. Destinations (tray, bowl) are not objects to move.
- excluded_objects: any object the original instruction says to leave, skip, ignore, or not touch. If none, use [].
- allowed_objects: objects that still need to be moved (not yet completed, not excluded).
- subtasks: one sub-task per allowed object that still needs action. Each sub-task is a single pick-and-place action.
- The tray (regardless of colour) is ALWAYS the destination. NEVER include it in visible_objects or allowed_objects.
- If an object from a completed sub-task is already on the tray, do NOT include it again.
- If the failed sub-task's object is still visible and not on the tray, include it in the revised plan.
- If the failure reason says the robot did not move or did not understand the instruction, the object name was probably too complex. Use the simplest possible name (e.g. "toy" not "stuffed animal", "cup" not "ceramic mug") or describe it by its most obvious visual feature (colour, shape).
- Keep every sub-task short and direct (e.g. "put the X on the tray").
- If the human operator provides additional guidance, treat it as ground truth about the scene and prefer it over your own visual interpretation wherever the two conflict.
"""


# ---------------------------------------------------------------------------
# Corrective note for a decomposition that left allowed objects unplanned
# ---------------------------------------------------------------------------
# Appended and the planner re-asked once when its own output contradicts itself:
# objects sit in allowed_objects (the work that remains) with no sub-task to do
# them. Names the gap rather than the rule, so the model has to resolve it.
UNCOVERED_OBJECTS_CONTEXT = """Your previous answer was INVALID. You listed these objects in allowed_objects but gave them no sub-task: {objects}

allowed_objects means "still needs to be moved", so every object in it MUST have exactly one sub-task. Answer again and resolve this for each of those objects, one of two ways:
- It still needs moving: keep it in allowed_objects and give it a sub-task.
- It needs no action (already at its destination, or the instruction says to leave it): move it to completed_objects or excluded_objects and take it OUT of allowed_objects.

Do not return an empty subtasks list while allowed_objects is non-empty."""


# ---------------------------------------------------------------------------
# User interjection during execution
# ---------------------------------------------------------------------------
class InterjectionType(enum.Enum):
    NONE = "none"
    SKIP = "skip"
    REPLAN = "replan"
    ABORT = "abort"


# Typed at any interjection prompt (and by the background listener) to abandon
# the whole high-level task and go back to asking for a new prompt.
ABORT_KEY = "q"


class InterjectionManager:
    """Background stdin listener that lets the user skip, replan, or abort
    mid-execution.

    In abort-only mode the listener still runs but honours nothing except
    ABORT_KEY, so 'q'+Enter always works even when skip/replan are disabled.
    """

    def __init__(self, client: "LoopRobotClient"):
        self.client = client
        self._type = InterjectionType.NONE
        self._replan_context = ""
        self._lock = threading.Lock()
        self._active = threading.Event()
        # Set while the main thread is blocking on stdin itself; the listener
        # must not consume input during that window.
        self._paused = threading.Event()
        self._abort_only = False
        self._thread: threading.Thread | None = None

    def start(self, abort_only: bool = False):
        self._clear()
        self._abort_only = abort_only
        self._active.set()
        self._thread = threading.Thread(target=self._listener, daemon=True)
        self._thread.start()

    def stop(self):
        self._active.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _clear(self):
        with self._lock:
            self._type = InterjectionType.NONE
            self._replan_context = ""

    def check_and_consume(self) -> tuple[InterjectionType, str]:
        """Return (type, context) and reset the interjection state."""
        with self._lock:
            t, ctx = self._type, self._replan_context
            self._type = InterjectionType.NONE
            self._replan_context = ""
        return t, ctx

    def request_abort(self):
        """Mark the high-level task as aborted and end any running episode."""
        with self._lock:
            self._type = InterjectionType.ABORT
            self._replan_context = ""
        self.client.episode_done.set()

    def _listener(self):
        if self._abort_only:
            print(f"\n[INTERJECT] Press '{ABORT_KEY}'+Enter to ABORT the whole task")
        else:
            print(f"\n[INTERJECT] Press 's'+Enter to SKIP subtask | 'r'+Enter to REPLAN "
                  f"| '{ABORT_KEY}'+Enter to ABORT the whole task | Enter to skip")
        while self._active.is_set():
            if self._paused.is_set():
                time.sleep(0.05)
                continue
            if select.select([sys.stdin], [], [], 0.2)[0]:
                # Re-check: the main thread may have claimed stdin while we
                # were blocked in select().
                if self._paused.is_set():
                    continue
                line = sys.stdin.readline().strip().lower()
                if not self._active.is_set():
                    break
                with self._lock:
                    abort_pending = self._type is InterjectionType.ABORT
                if abort_pending and line != ABORT_KEY:
                    # An abort is already queued and not yet consumed; a stray
                    # keystroke must not downgrade it to a skip/replan.
                    continue
                if line == ABORT_KEY:
                    print("[INTERJECT] ABORT requested — stopping current action...")
                    self.request_abort()
                elif self._abort_only:
                    # Skip/replan are disabled; ignore everything but the abort
                    # key rather than acting on a keystroke the operator opted out of.
                    continue
                elif line == "r":
                    # End the episode immediately so the arm stops before the
                    # main thread replans; the robot never keeps executing into
                    # a replan.
                    with self._lock:
                        self._type = InterjectionType.REPLAN
                        self._replan_context = ""
                    print("[INTERJECT] REPLAN requested — stopping current action...")
                    self.client.episode_done.set()
                else:
                    # 's', bare Enter, or anything else → skip
                    with self._lock:
                        self._type = InterjectionType.SKIP
                    print("[INTERJECT] SKIP requested — stopping current action...")
                    self.client.episode_done.set()


# ---------------------------------------------------------------------------
# Output parsing helpers
# ---------------------------------------------------------------------------
def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences the model sometimes wraps JSON in."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def is_fixture(name: str) -> bool:
    """A name for part of the setup (tray, table, robot arm, wires, label, ...) rather than
    something to move — SURVEY_FIXTURE_WORDS, as remaining_from_survey uses."""
    return any(re.search(rf"\b{w}", str(name).lower()) for w in SURVEY_FIXTURE_WORDS)


def planner_objects(names: list[str]) -> tuple[list[str], list[str]]:
    """(objects for the planner, dropped fixture names) from an identified-object list.

    Same shape as a targeted retry round hands the planner (remaining + destinations +
    ["table"]): fixtures are removed, then the tray(s) and the table come back as
    destinations, so the arm, its wires or its label can never be planned as objects.
    """
    objects = [n for n in names if not is_fixture(n)]
    trays = [n for n in names if is_fixture(n) and "tray" in str(n).lower()] or ["tray"]
    dropped = [n for n in names if is_fixture(n) and n not in trays and str(n).lower() != "table"]
    return objects + trays + ["table"], dropped


def drop_fixture_subtasks(tasks: list[str]) -> tuple[list[str], list[str]]:
    """(kept, dropped): sub-tasks whose moved object is a fixture (the tray, the table, a
    robot part). The moved object is the text before the destination ("put the tray on
    the table" -> "put the tray"), so "put the banana on the tray" is kept."""
    def moved(task: str) -> str:
        return re.split(r"\s+(?:on|onto|in|into|inside|next to|to)\s+", task.lower(), maxsplit=1)[0]
    bad = [t for t in tasks if is_fixture(moved(t))]
    return [t for t in tasks if t not in bad], bad


def parse_decomposition(output: str) -> tuple[list[str], dict]:
    """Parse a decomposition into (sub-tasks, the full object dict).

    The dict carries visible/excluded/completed/allowed_objects, which the
    caller needs to check that every object the planner said still needs moving
    actually got a sub-task. Empty dict when the model returned a bare list.
    """
    text = _strip_code_fences(output.strip())

    try:
        result = json.loads(text)
        if isinstance(result, dict) and "subtasks" in result:
            tasks = result["subtasks"]
            print(f"\nVisible: {result.get('visible_objects')}")
            print(f"Excluded: {result.get('excluded_objects')}")
            print(f"Completed: {result.get('completed_objects')}")
            print(f"Allowed: {result.get('allowed_objects')}")
        else:
            tasks = result
            result = {}
        print(f"\nParsed {len(tasks)} sub-tasks:")
        for i, task in enumerate(tasks, 1):
            print(f"  {i}. {task}")
        return tasks, result
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse subtasks from VLM output:\n{output}")


def parse_subtask_list(output: str) -> list[str]:
    return parse_decomposition(output)[0]


def uncovered_allowed_objects(tasks: list[str], parsed: dict) -> list[str]:
    """Objects the planner listed as still needing a move but gave no sub-task.

    A non-empty result means the plan contradicts itself: `allowed_objects` is
    by definition the work that remains, so every entry must appear in some
    sub-task. Matching is substring-based because sub-tasks name objects in
    prose ("put the pouch on the tray").
    """
    allowed = parsed.get("allowed_objects")
    if not isinstance(allowed, list):
        return []
    joined = " ".join(str(t) for t in tasks).lower()
    return [str(o) for o in allowed if str(o).lower() not in joined]


def parse_evaluation(output: str) -> dict:
    """Extract a {'success': bool, 'reason': str} object from model output."""
    text = _strip_code_fences(output)

    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "success" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not parse evaluation from VLM output:\n{output}")


# ---------------------------------------------------------------------------
# VLM planner: wraps decomposition, evaluation, and replanning
# ---------------------------------------------------------------------------
class VLMPlanner:
    """Holds the loaded VLM and exposes decompose(), evaluate(), and replan()."""

    def __init__(self, model_name: str, temperature: float = 0.7,
                 eval_fps: float = 2.0, two_pass: bool = True,
                 reask_uncovered: bool = False, filter_fixtures: bool = True):
        self.model, self.processor = load_model(model_name)
        self.temperature = temperature
        self.eval_fps = eval_fps
        # Two-pass decomposition: identify objects, then decompose given the list.
        self.two_pass = two_pass
        # Re-ask the planner once when allowed_objects has an object with no
        # sub-task. Off by default: on first-round plans it never recovered a
        # dropped object and inflated every count-limited instruction ("put one
        # object on the tray") into moving everything (vlm_decompose_test, 39
        # scenes: 79 re-asks, 70 broke a correct plan, 0 clean fixes).
        self.reask_uncovered = reask_uncovered
        # Two-pass only: fixtures (SURVEY_FIXTURE_WORDS) leave the identified objects
        # before planning — the tray and table come back as destinations, as in a
        # targeted retry — and any sub-task that would move a fixture is dropped.
        self.filter_fixtures = filter_fixtures

    def _user_content(self, frame: Image.Image | None, text: str) -> list[dict]:
        content = []
        if frame is not None:
            content.append({"type": "text", "text": "[top-down camera]"})
            content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": text})
        return content

    def _user_content_video(self, clip: list[Image.Image], text: str,
                            fps: float | None = None) -> list[dict]:
        return [
            {"type": "text", "text": "[top-down camera, video of the attempt]"},
            {"type": "video", "video": clip, "fps": fps or self.eval_fps},
            {"type": "text", "text": text},
        ]

    def decompose(self, prompt: str, frame: Image.Image | None,
                  extra_context: str = "",
                  objects: list[str] | None = None) -> list[str]:
        """High-level prompt + observation -> ordered list of sub-tasks.

        Two-pass (default) identifies the visible objects first, then decomposes
        given that list; both calls share the single `frame`. Falls back to the
        single-call prompt when two-pass is disabled or there is no frame to
        identify objects from.

        `extra_context` is appended after the instruction — used on a re-decompose
        to tell the planner why the previous attempt was judged incomplete.

        `objects`, when given, replaces the identification pass: the planner is
        told these are the visible objects (a targeted retry round passes the
        final-eval survey's objects that are not yet at a destination).
        """
        if self.two_pass and frame is not None:
            image_content = [
                {"type": "text", "text": "[top-down camera]"},
                {"type": "image", "image": frame},
            ]
            if objects is not None:
                visible = list(objects)
                logging.info(f"Objects given to the planner ({len(visible)}): {visible}")
            else:
                visible = identify_objects(
                    self.model, self.processor, image_content, self.temperature
                )
                logging.info(f"Identified objects ({len(visible)}): {visible}")
                if self.filter_fixtures:
                    visible, dropped = planner_objects(visible)
                    if dropped:
                        logging.info(f"Fixtures removed from the object list: {dropped}")
            output = decompose_from_objects(
                self.model, self.processor, image_content, prompt, visible,
                self.temperature, extra_context=extra_context,
            )
            logging.debug(f"Raw VLM decomposition output ({len(output)} chars): {output!r}")
            tasks, parsed = parse_decomposition(output)

            # Optional (reask_uncovered): the planner is required to give every
            # object in allowed_objects a sub-task. When it does not, name the gap
            # and give it one chance to fix it. Re-asking rather
            # than synthesising "put the X on the tray" keeps destination
            # inference in the model, which arrangement tasks depend on.
            missing = uncovered_allowed_objects(tasks, parsed) if self.reask_uncovered else []
            if missing:
                logging.warning(
                    f"Decomposition left {len(missing)} allowed object(s) with no "
                    f"sub-task: {missing} — re-asking the planner once."
                )
                retry_note = UNCOVERED_OBJECTS_CONTEXT.format(
                    objects=", ".join(missing)
                )
                output = decompose_from_objects(
                    self.model, self.processor, image_content, prompt, visible,
                    self.temperature,
                    extra_context=(f"{extra_context}\n\n{retry_note}"
                                   if extra_context else retry_note),
                )
                logging.debug(f"Raw VLM re-decomposition output ({len(output)} chars): {output!r}")
                tasks, parsed = parse_decomposition(output)
                still_missing = uncovered_allowed_objects(tasks, parsed)
                if still_missing:
                    logging.warning(
                        f"Re-ask still left {still_missing} uncovered; proceeding "
                        f"with {len(tasks)} sub-task(s)."
                    )
            if self.filter_fixtures:
                tasks, dropped = drop_fixture_subtasks(tasks)
                if dropped:
                    logging.warning(f"Dropped sub-task(s) moving a fixture: {dropped}")
            return tasks
        else:
            text = f"\nInstruction: {prompt}"
            if frame is None:
                text = (
                    "No camera observation is available. Decompose the instruction "
                    "based on the text alone.\n" + text
                )
            if extra_context:
                text += f"\n\n{extra_context}"
            messages = [
                {"role": "system", "content": [{"type": "text", "text": DECOMPOSITION_SYSTEM_PROMPT}]},
                {"role": "user", "content": self._user_content(frame, text)},
            ]
            output = generate(
                self.model, self.processor, messages, temperature=self.temperature
            )
        logging.debug(f"Raw VLM decomposition output ({len(output)} chars): {output!r}")
        return parse_subtask_list(output)

    def evaluate(
        self,
        sub_task: str,
        observation: "Image.Image | list[Image.Image] | None",
        fps: float | None = None,
        extra_context: str | None = None,
    ) -> dict:
        """Observation + attempted sub-task -> {'success': bool, 'reason': str}.

        `observation` may be a single PIL frame (still) or a list of PIL frames
        (a video clip of the attempt). An empty list is treated as no observation.

        `fps` overrides the configured nominal rate — pass the clip's true rate
        (frames / episode duration) so the model's frame timestamps span the
        real length of the attempt.

        `extra_context` is appended to the prompt as additional information for
        the judgement — e.g. the reason the continuous monitor stopped the episode
        early, so the evaluator re-judges the attempt informed by why it was cut short.
        """
        text = (
            f'\nThe robot just attempted this sub-task: "{sub_task}"\nDid it succeed?'
        )
        if extra_context:
            text += (
                "\n\nThe episode was stopped early by a continuous monitor for this "
                f"reason: {extra_context}\nTake this into account in your judgement."
            )
        if isinstance(observation, list) and observation:
            content = self._user_content_video(observation, text, fps=fps)
        else:
            frame = observation if isinstance(observation, Image.Image) else None
            content = self._user_content(frame, text)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": EVALUATION_SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        output = generate(
            self.model, self.processor, messages, temperature=self.temperature
        )
        logging.debug(f"Raw eval output: {output}")
        return parse_evaluation(output)

    def evaluate_final(
        self,
        original_prompt: str,
        frame: "Image.Image | None",
        initial_frame: "Image.Image | None" = None,
    ) -> dict:
        """Final scene + original instruction -> {'success': bool, 'reason': str}.

        Judges the whole high-level instruction from a single still of the final
        scene, with no knowledge of what the pipeline believes it completed.

        Two calls by default (survey -> judge). The survey sees the image but not
        the instruction; the judge sees the instruction but only the surveyed
        locations. That split is what stops the evaluator asserting an object
        reached the tray because the instruction asked for it — it no longer
        writes the location it judges against. Falls back to the single-call
        prompt when there is no frame, two-pass is off, or the survey comes back
        empty.

        With `initial_frame` (the scene before the robot started) the BEFORE/AFTER
        path runs instead — see _evaluate_final_before_after — falling back to the
        paths above if its survey comes back empty.
        """
        if self.two_pass and frame is not None and initial_frame is not None:
            verdict = self._evaluate_final_before_after(original_prompt, frame, initial_frame)
            if verdict is not None:
                return verdict
            logging.warning("BEFORE/AFTER survey returned no rows — falling back to the "
                            "single-image final evaluation.")
        if self.two_pass and frame is not None:
            image_content = [
                {"type": "text", "text": "[top-down camera]"},
                {"type": "image", "image": frame},
            ]
            surveyed = survey_scene(
                self.model, self.processor, image_content, self.temperature
            )
            if surveyed:
                # Row count is logged because a thin survey is the one failure
                # this design cannot catch: an object the survey never mentions
                # is invisible to the judge, which only sees these rows.
                logging.info(f"Final-eval surveyed {len(surveyed)} object(s):")
                for row in surveyed:
                    logging.info(f"  {row.get('object')} -> {row.get('support')} "
                                 f"({row.get('evidence')})")
                verdict = self._judge_final(original_prompt, surveyed)
                # Handed back so a retry round can plan from what the survey saw.
                verdict["surveyed"] = surveyed
                return verdict
            logging.warning(
                "Final-eval scene survey returned no rows — falling back to the "
                "single-call final evaluation."
            )

        text = (
            f'\nThe robot was given this high-level instruction: "{original_prompt}"'
            f'\nLooking at the final scene, was the whole instruction accomplished?'
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": FINAL_EVALUATION_SYSTEM_PROMPT}]},
            {"role": "user", "content": self._user_content(frame, text)},
        ]
        output = generate(
            self.model, self.processor, messages, temperature=self.temperature
        )
        logging.debug(f"Raw final-eval output: {output}")
        return parse_evaluation(output)

    def _requirements(self, instruction: str, findings: list[dict],
                      before: "Image.Image") -> dict:
        """Which listed objects the instruction requires on / off the tray. Sees the
        instruction, the object names and the BEFORE frame — never where anything ended
        up. ValueError if the output does not parse."""
        text = (f"Objects on the table: {json.dumps([f['object'] for f in findings])}"
                f'\nInstruction: "{instruction}"')
        content = [
            {"type": "text", "text": "[BEFORE — the table before the robot started. Use it only "
                                     "to see what each listed object is (colour, material, kind); "
                                     "it does not show where anything is now.]"},
            {"type": "image", "image": before},
            {"type": "text", "text": text},
        ]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": FINAL_REQUIREMENTS_SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        output = generate(self.model, self.processor, messages, temperature=self.temperature)
        logging.debug(f"Raw requirements output: {output}")
        return parse_requirements(output)

    def precheck_subtask_done(self, sub_task: str, before: "Image.Image",
                              after: "Image.Image") -> dict:
        """Is the sub-task's object on the tray at the end of the attempt?

        BEFORE/AFTER survey of the clip's first and last frames (no instruction), stacks
        resolved in code, the requirements call on the sub-task text to find which
        listed object it means, then code checks that object's resolved location.
        Returns {done, reason, targets, findings, surveyed, requirements}. `done` is
        False whenever the object cannot be identified — the evaluator then decides.
        """
        surveyed = survey_scene_change(self.model, self.processor, before, after,
                                       self.temperature)
        out = dict(done=False, targets=[], findings=[], surveyed=surveyed, requirements=None)
        if not surveyed:
            return dict(out, reason="survey returned no rows")
        findings = resolve_findings(surveyed)
        req = self._requirements(sub_task, findings, before)
        sets = compare_requirements(req, findings)["sets"]
        where = {f["object"].lower(): f["support"].split(" (")[0] for f in findings}
        targets = sets["must_be_on_tray"] + sets["choose_from"]
        out.update(findings=findings, requirements=req, targets=targets)
        if not targets:
            return dict(out, reason=f"no listed object matches the sub-task ({list(where)})")
        placed = [t for t in targets if where[t] == "tray"]
        need = sets["choose_count"] if sets["choose_from"] and not sets["must_be_on_tray"] \
            else len(targets)
        done = len(placed) >= max(need, 1)
        return dict(out, done=done,
                    reason=(f"{', '.join(placed)} already on the tray" if done else
                            f"{', '.join(f'{t} ({where[t]})' for t in targets)} not on the tray"))

    def _evaluate_final_before_after(self, original_prompt: str, frame: "Image.Image",
                                     initial_frame: "Image.Image") -> dict | None:
        """BEFORE/AFTER survey -> stacks resolved in code -> requirements from the
        instruction + object names + BEFORE frame -> code comparison.

        Returns {success, reason, surveyed, findings, requirements, sets}, or None when
        the survey has no rows. Raises ValueError if the requirements do not parse
        (the caller then has no verdict, as for any unparseable final evaluation).
        """
        surveyed = survey_scene_change(self.model, self.processor, initial_frame, frame,
                                       self.temperature)
        if not surveyed:
            return None
        findings = resolve_findings(surveyed)
        logging.info(f"Final-eval BEFORE/AFTER survey, {len(findings)} object(s):")
        for f in findings:
            logging.info(f"  {f['object']}: {f['before']} -> {f['support']}")
        req = self._requirements(original_prompt, findings, initial_frame)
        verdict = compare_requirements(req, findings)
        s = verdict["sets"]
        logging.info(f"Final-eval requirements: on tray {s['must_be_on_tray']}, off tray "
                     f"{s['must_not_be_on_tray']}"
                     + (f", {s['choose_count']} of {s['choose_from']}" if s["choose_from"] else "")
                     + (f" | unmatched names {s['unmatched']}" if s["unmatched"] else ""))
        verdict.update(surveyed=surveyed, findings=findings, requirements=req)
        return verdict

    def _judge_final(self, original_prompt: str, located: list[dict]) -> dict:
        """Judging half of the two-pass final evaluation.

        Text-only: the perception rows stand in for the image, so the judge can
        compare against the instruction without being able to restate where
        anything is. Emits the same {success, reason} contract as the
        single-call path, so `parse_evaluation` is unchanged.
        """
        findings = [
            {"object": str(r.get("object")), "support": str(r.get("support"))}
            for r in located
        ]
        text = (
            f'The robot was given this high-level instruction: "{original_prompt}"'
            f"\n\nPerception findings for the final scene:"
            f"\n{json.dumps(findings)}"
            f"\n\nWas the whole instruction accomplished?"
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": FINAL_JUDGE_SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ]
        output = generate(
            self.model, self.processor, messages, temperature=self.temperature
        )
        logging.debug(f"Raw final-judge output: {output}")
        return parse_evaluation(output)

    def refine_label(
        self,
        failed_instruction: str,
        failure_reason: str,
        frame: "Image.Image | None" = None,
    ) -> list[str]:
        """Suggest alternative object names after a vocabulary failure."""
        text = (
            f'The robot failed to execute: "{failed_instruction}"\n'
            f"Diagnosis: {failure_reason}\n\n"
            "Suggest 3 alternative names for the target object."
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": REFINEMENT_SYSTEM_PROMPT}]},
            {"role": "user", "content": self._user_content(frame, text)},
        ]
        output = generate(
            self.model, self.processor, messages, temperature=self.temperature
        )
        logging.debug(f"Raw refinement output: {output}")
        lines = [
            re.sub(r"^[\d.\-\)\s]+", "", line).strip()
            for line in output.strip().splitlines()
        ]
        return [l for l in lines if l]

    def replan(
        self,
        original_prompt: str,
        completed_subtasks: list[str],
        failed_subtask: str,
        failure_reason: str,
        frame: Image.Image | None,
    ) -> list[str]:
        """Replan remaining sub-tasks given what succeeded, what failed, and the
        current scene state."""
        context_parts = [f'Original instruction: "{original_prompt}"']

        if completed_subtasks:
            completed_str = "\n".join(
                f"  {i}. {st}" for i, st in enumerate(completed_subtasks, 1)
            )
            context_parts.append(f"Completed sub-tasks (do NOT repeat):\n{completed_str}")
        else:
            context_parts.append("No sub-tasks have been completed yet.")

        context_parts.append(
            f'Failed sub-task: "{failed_subtask}"\n'
            f'Failure reason: {failure_reason}'
        )

        context_parts.append(
            "Produce a revised sub-task list for the REMAINING work only."
        )

        text = "\n\n".join(context_parts)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": REPLAN_SYSTEM_PROMPT}]},
            {"role": "user", "content": self._user_content(frame, text)},
        ]
        output = generate(
            self.model, self.processor, messages, temperature=self.temperature
        )
        logging.debug(f"Raw VLM replan output ({len(output)} chars): {output!r}")
        return parse_subtask_list(output)


# ---------------------------------------------------------------------------
# Frame source for the VLM
# ---------------------------------------------------------------------------
class VLMFrameSource:
    """Provides PIL frames for the VLM from the robot's camera observations."""

    def __init__(self, client: LoopRobotClient, camera_key: str,
                 session_dir: Path, save_frames: bool):
        self.client = client
        self.camera_key = camera_key
        # session_dir is the per-run root (always set). task_dir is the
        # currently-active sub-folder, repointed for each high-level task. Frame
        # PNGs are only written when save_frames is True, but the per-task dir
        # always exists so the task log has a home either way.
        self.session_dir = session_dir
        self.task_dir = session_dir
        self.save_frames = save_frames
        self._task_counter = 0

    def start_task(self) -> Path:
        """Begin a new high-level task: route subsequent saves into a fresh
        `NN` sub-folder under the session dir, create it, and return it so the
        caller can co-locate the task log there."""
        self._task_counter += 1
        self.task_dir = self.session_dir / f"{self._task_counter:02d}"
        self.task_dir.mkdir(parents=True, exist_ok=True)
        return self.task_dir

    def capture(self, tag: str = "frame") -> Image.Image | None:
        obs = self.client.capture_frame()

        # Try the specified camera key first
        frame = None
        if self.camera_key in obs:
            value = obs[self.camera_key]
            if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] == 3:
                frame = Image.fromarray(value.astype(np.uint8))

        # Fallback: first image-like array
        if frame is None:
            for key, value in obs.items():
                if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] == 3:
                    logging.warning(f"Camera key '{self.camera_key}' not found; "
                                    f"using '{key}' instead.")
                    frame = Image.fromarray(value.astype(np.uint8))
                    break

        if frame is None:
            logging.warning("No VLM frame available — running text-only.")
            return None

        if self.save_frames:
            self.task_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = self.task_dir / f"{tag}_{ts}.png"
            frame.save(path)
            logging.info(f"Saved VLM frame: {path}")

        return frame

    def capture_clip(self, tag: str = "clip",
                     num_frames: int = 8) -> tuple[list[Image.Image], float]:
        """Return the buffered episode clip as PIL frames (oldest-first, evenly
        spaced in time across the whole episode) plus the wall-clock span they cover.

        The span comes from the client's frame timestamps. The buffer decimates
        rather than dropping its head, so it spans the full episode and the span is
        close to the episode duration. Returns ([], 0.0) if the client buffered
        nothing (e.g. a very short episode), in which case the caller should fall
        back to a single still.
        """
        arrays, span = self.client.get_episode_clip(num_frames)
        clip: list[Image.Image] = []
        for arr in arrays:
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] == 3:
                clip.append(Image.fromarray(arr.astype(np.uint8)))

        if not clip:
            logging.warning("Episode clip buffer empty — no video for evaluation.")
            return [], 0.0

        if self.save_frames:
            self.task_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = self.task_dir / f"{tag}_{ts}_recap.png"
            self._render_clip_recap(clip, span, path)
            logging.info(f"Saved VLM eval recap ({len(clip)} frames): {path}")

        return clip, span

    def _render_clip_recap(self, clip: list[Image.Image], span: float,
                           path: Path) -> None:
        """Render the eval clip as a single annotated contact sheet.

        Mirrors robot_client_loop._render_episode_recap: each frame becomes a tile
        labelled with its elapsed time (derived by evenly spacing across `span`,
        since the clip is already sampled), and tiles are pasted into a near-square
        grid on a dark canvas. Wrapped so a recap failure never breaks the run.
        """
        try:
            from PIL import ImageDraw

            n = len(clip)
            tiles = []
            for i, frame in enumerate(clip):
                img = frame.convert("RGB")
                elapsed = span * i / (n - 1) if n > 1 else 0.0
                label = f"t={elapsed:4.1f}s #{i}"
                draw = ImageDraw.Draw(img)
                draw.rectangle([0, 0, img.width, 15], fill=(0, 0, 0))
                draw.text((2, 2), label, fill=(255, 255, 255))
                tiles.append(img)

            cols = max(1, int(np.ceil(np.sqrt(n))))
            n_rows = int(np.ceil(n / cols))
            tw, th = tiles[0].size
            sheet = Image.new("RGB", (cols * tw, n_rows * th), (30, 30, 30))
            for i, tile in enumerate(tiles):
                sheet.paste(tile, ((i % cols) * tw, (i // cols) * th))
            sheet.save(path)
        except Exception as e:  # noqa: BLE001 - never let recap break a run
            logging.error(f"Failed to render eval clip recap: {e}")

    def stop(self):
        pass  # No separate pipeline to clean up


# ---------------------------------------------------------------------------
# Orchestration: one high-level prompt -> decompose -> execute -> evaluate
# ---------------------------------------------------------------------------
@dataclass
class RoundResult:
    """Outcome of one decompose -> execute -> final-evaluate round.

    `stopped` marks a round that ended for a reason no further round can fix (the
    operator aborted, decomposition failed to parse, the evaluator errored out) —
    the caller must not start another round. `final_success` is None when no
    overall verdict was produced (--evaluate_final=false, or the final evaluation
    failed to parse/errored), which likewise ends the task.
    """
    stopped: bool = False
    final_success: bool | None = None
    final_reason: str = ""
    planned: int = 0  # sub-tasks this round's decomposition produced
    # Sub-tasks this round finished — i.e. what it believes it accomplished, in
    # execution order. Carried out of the round so a retry re-decompose can tell
    # the planner which work is already done instead of re-deriving it from the
    # image alone.
    completed: list[str] = field(default_factory=list)
    # The final evaluation's scene survey rows ({object, support, evidence});
    # empty when the single-call final eval ran or the survey came back empty.
    final_survey: list[dict] = field(default_factory=list)
    # The scene before the robot started (round 1's decompose frame), handed to every
    # later round's final evaluation as its BEFORE image.
    initial_frame: "Image.Image | None" = None


# Handed to the planner when re-decomposing after a failed final evaluation.
RETRY_DECOMPOSE_CONTEXT = """A previous attempt at this same instruction was judged INCOMPLETE by the evaluator, who reported:
{reason}
{completed}
The camera image shows the CURRENT scene, after that attempt. The image is GROUND TRUTH: where the evaluator's report and the image disagree, believe the image.

Plan only the work that still REMAINS to satisfy the original instruction. If an object is already at its destination in the image, list it in completed_objects — do not put it in allowed_objects and do not give it a sub-task. Do not move any object the instruction says to leave alone; those go in excluded_objects.

Remember the invariant: every object in allowed_objects gets exactly one sub-task. Never return an empty subtasks list while allowed_objects is non-empty — an object that needs no action belongs in completed_objects."""


# Slotted into RETRY_DECOMPOSE_CONTEXT's {completed}.
COMPLETED_CONTEXT = """
These sub-tasks were already executed and verified complete in the previous attempt:
{lines}
The objects they name are already at their destinations. Put those objects in completed_objects, NOT in allowed_objects, and do not plan any sub-task for them.
"""

NOTHING_COMPLETED_CONTEXT = """
No sub-task from the previous attempt completed successfully, so treat none of the work as done on that basis. Judge from the image alone which objects, if any, are already at their destination.
"""


# Appended to the retry context when a targeted retry round hands the planner the
# survey's objects instead of a fresh identify pass.
TARGETED_RETRY_NOTE = """The visible_objects list for this attempt comes from a survey of the current scene. It contains ONLY the objects that are NOT yet at a destination, plus the destinations themselves. Decide which of these objects the original instruction still requires moving, and plan only those. Objects the instruction says to leave go in excluded_objects. If none of them needs moving, return empty allowed_objects and an empty subtasks list — that is a valid answer here."""


# Survey rows that name part of the setup rather than something to move. The
# survey prompt already forbids these, but it still emits them now and then
# ("pink tray", "orange robot arm", "hand").
SURVEY_FIXTURE_WORDS = ("tray", "table", "robot", "arm", "gripper", "hand", "wire",
                        "cable", "label", "surface", "camera", "cord")


# ---------------------------------------------------------------------------
# BEFORE/AFTER final evaluation: requirements from the instruction, compared in code
# ---------------------------------------------------------------------------
# The requirements call never sees where anything ended up (only the instruction, the
# object names and the BEFORE frame). Given the final locations, a judge fits "what
# was required" to the scene it sees; this split is what stops that.
FINAL_REQUIREMENTS_SYSTEM_PROMPT = """You decide what a tabletop robot instruction requires. You receive the instruction and the names of the objects on the table. You do NOT see the scene and are not told where anything is — decide only what the instruction asks for.

The tray is the destination: "the tray" and "away" always mean the tray. The robot arm, the table and the tray are not objects.

Sort every listed object into exactly one group:
- must_be_on_tray: the instruction asks for it to end up on the tray.
- must_not_be_on_tray: the instruction asks to leave it, skip it, keep it or not touch it — or simply does not ask for it to be moved. An object the instruction does not ask to move must stay off the tray.
- choose_from, with choose_count: only when the instruction lets the robot choose ("put one object on the tray", "put two of the brown objects on the tray") — the objects it may choose from, and exactly how many it must place.

Rules:
- Names may differ from the instruction's words: the instruction's "pouch" may be listed as a "brown cylinder" or "brown roll", its "toy" as a "stuffed animal". Match by what each object plausibly is. For words about colour, material or kind ("brown", "soft", "edible", "fruit"), decide from what each named object is.
- Words about order — first, then, next, last, after, starting with — only describe the order of placement. Ignore order: an object that "goes last" is placed after the others, so it must still end up on the tray.
- Conditions ("if there is an apple on the table ...") are decided from the list: an object that is not listed is not on the table.
- Use the names exactly as listed, and put every listed object in exactly one group.

Return ONLY JSON:
{"must_be_on_tray": ["<name>", ...], "must_not_be_on_tray": ["<name>", ...], "choose_from": ["<name>", ...], "choose_count": <number>}

Examples, for the objects ["apple", "red cup", "banana"]:
Instruction: "put everything on the tray except the banana"
{"must_be_on_tray": ["apple", "red cup"], "must_not_be_on_tray": ["banana"], "choose_from": [], "choose_count": 0}
Instruction: "put one of the red objects on the tray"
{"must_be_on_tray": [], "must_not_be_on_tray": ["banana"], "choose_from": ["apple", "red cup"], "choose_count": 1}
Instruction: "put the fruit on the tray, the banana first"
{"must_be_on_tray": ["apple", "banana"], "must_not_be_on_tray": ["red cup"], "choose_from": [], "choose_count": 0}
Instruction: "put everything on the tray, but the red cup goes last"
{"must_be_on_tray": ["apple", "red cup", "banana"], "must_not_be_on_tray": [], "choose_from": [], "choose_count": 0}
"""


def resolve_findings(rows: list[dict]) -> list[dict]:
    """Survey rows -> {object, before, support} with each support resolved to where the
    object ends up.

    A support naming another surveyed object (by the same, a shorter or a longer
    name) is followed down the stack to a surface, so the toy on the banana in the
    tray becomes "tray (on top of the banana)". One finding per object: a repeated
    object (the survey sometimes lists BEFORE rows, then AFTER rows) keeps its last row.
    """
    names = [str(r.get("object", "")).strip() for r in rows]
    support = {n.lower(): str(r.get("support", "")).strip().lower()
               for n, r in zip(names, rows) if n}

    def as_object(sup: str, me: str) -> str | None:
        if not sup or "tray" in sup or "table" in sup:
            return None  # a surface, never another object
        if sup in support and sup != me:
            return sup
        hits = [k for k in support if k != me and (sup in k or k in sup)]
        return max(hits, key=len) if hits else None

    def settle(key: str) -> str:
        seen = {key}
        while (under := as_object(support[key], key)) and under not in seen:
            seen.add(under)
            key = under
        sup = support[key]
        return "tray" if "tray" in sup else "table" if "table" in sup else sup or "unknown"

    findings, last = [], {n.lower(): i for i, n in enumerate(names) if n}
    for i, (n, r) in enumerate(zip(names, rows)):
        if not n or last[n.lower()] != i:
            continue
        under = as_object(support[n.lower()], n.lower())
        how = "hidden under" if "hidden" in str(r.get("evidence", "")).lower() else "on top of"
        findings.append({"object": n, "before": str(r.get("before")),
                         "support": settle(n.lower()) + (f" ({how} the {under})" if under else "")})
    return findings


def parse_requirements(output: str) -> dict:
    """The requirements call's JSON object; ValueError if there is none."""
    text = _strip_code_fences(output)
    for cand in (text, text[text.find("{"): text.rfind("}") + 1]):
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse final-eval requirements from VLM output:\n{output}")


def compare_requirements(req: dict, findings: list[dict]) -> dict:
    """Code verdict: the required sets against the resolved locations.

    Names snap onto the findings (exact, then containment). An object the call listed
    nowhere must stay off the tray (not asked to move -> leave it). If it listed an
    object both as allowed and as must-not, the allowed group wins — "put one object"
    tends to come back with every object in both. Returns {success, reason, sets}.
    """
    names = [f["object"] for f in findings]
    where = {f["object"].lower(): f["support"].split(" (")[0] for f in findings}

    def snap(n: str) -> str | None:
        n = str(n).strip().lower()
        if n in where:
            return n
        hits = [k for k in where if n and (n in k or k in n)]
        return max(hits, key=len) if hits else None

    groups, unmatched = {}, []
    for key in ("must_be_on_tray", "must_not_be_on_tray", "choose_from"):
        vals = req.get(key) or []
        if not isinstance(vals, list):
            raise ValueError(f"requirements field {key} is not a list: {vals!r}")
        groups[key] = []
        for v in vals:
            k = snap(v)
            (groups[key].append(k) if k else unmatched.append(str(v)))
    allowed = set(groups["must_be_on_tray"]) | set(groups["choose_from"])
    groups["must_not_be_on_tray"] = [k for k in groups["must_not_be_on_tray"] if k not in allowed]
    listed = {k for g in groups.values() for k in g}
    unlisted = [n for n in names if n.lower() not in listed]
    groups["must_not_be_on_tray"] += [n.lower() for n in unlisted]
    count = int(req.get("choose_count") or 0)

    problems = [f"{k} should be on the tray (now: {where[k]})"
                for k in groups["must_be_on_tray"] if where[k] != "tray"]
    problems += [f"{k} should not be on the tray" for k in groups["must_not_be_on_tray"]
                 if where[k] == "tray"]
    if groups["choose_from"]:
        placed = [k for k in groups["choose_from"] if where[k] == "tray"]
        if len(placed) != count:
            problems.append(f"{len(placed)} of {groups['choose_from']} on the tray, "
                            f"expected exactly {count}")
    return {"success": not problems,
            "reason": "; ".join(problems) or "every object is where the instruction requires",
            "sets": {**groups, "choose_count": count, "unlisted": unlisted, "unmatched": unmatched}}


def remaining_from_survey(rows: list[dict]) -> tuple[list[str], list[str]]:
    """Split a final-eval survey into (objects not at a destination, destinations).

    Each row names an object and what it rests on. A support that is not itself a
    surveyed object is a surface or container (table, tray, box). An object counts
    as placed when its support chain ends on a container other than the table —
    a banana lying on the pouch that sits in the tray is placed. Everything else
    still rests on the table (or on something that does) and is returned for the
    planner to decide on. Fixture rows are dropped.
    """
    support = {}
    for r in rows:
        name = str(r.get("object", "")).strip()
        if name and not is_fixture(name):
            support[name.lower()] = (name, str(r.get("support", "")).strip().lower())

    def as_object(sup: str) -> str:
        """A support naming a surveyed object, possibly by a shorter or longer
        name ("pouch" for the "brown pouch" row), resolves to that object."""
        if sup in support:
            return sup
        return next((k for k in support if sup and (sup in k or k in sup)), sup)

    support = {k: (name, as_object(sup)) for k, (name, sup) in support.items()}
    # "missing" (the BEFORE/AFTER survey's answer for an object it cannot find) is not
    # a container: that object is not placed.
    containers = sorted({sup for _, sup in support.values()
                         if sup and sup not in support and "table" not in sup
                         and sup not in ("missing", "unknown")})

    def placed(key: str) -> bool:
        seen = set()
        while key in support and key not in seen:
            seen.add(key)
            key = support[key][1]
        return bool(key) and key in containers

    remaining = [name for key, (name, _) in support.items() if not placed(key)]
    # The tray is this setup's fixed destination; an empty tray supports nothing,
    # so the survey never names it — add it so the planner still has somewhere
    # to put things.
    destinations = containers if any("tray" in c for c in containers) else containers + ["tray"]
    return remaining, destinations


def _format_completed_context(completed: list[str]) -> str:
    """Render the completed sub-tasks as the {completed} block of the retry context."""
    if not completed:
        return NOTHING_COMPLETED_CONTEXT
    lines = "\n".join(f"- {st}" for st in completed)
    return COMPLETED_CONTEXT.format(lines=lines)


def run_high_level_task(
    client: LoopRobotClient,
    planner: VLMPlanner,
    frames: VLMFrameSource,
    cfg: OrchestratorConfig,
    prompt: str,
    interjection: InterjectionManager,
    log_sink: "TaskLogSink",
):
    logger = client.logger

    # Route this task's saved frames into a fresh session/<task> sub-folder, and
    # mirror all terminal output for this task into task.log inside that folder.
    # Per-action joint targets are far too dense for task.log (and for the
    # terminal), so they go to actions.log beside it. Every round of this task
    # shares the one folder and the one log.
    task_dir = frames.start_task()
    log_sink.open(task_dir / "task.log")
    client.start_action_log(task_dir / "actions.log")
    try:
        retries_left = max(cfg.max_final_retries, 0)
        round_num = 0
        retry_context = ""
        retry_objects: list[str] | None = None
        # Union of every round's completed sub-tasks, in first-completed order.
        # Accumulated rather than taken from the last round alone so that with
        # --max_final_retries > 1 round 3 still knows what round 1 finished.
        completed_all: list[str] = []
        # Round 1's decompose frame: the BEFORE image for every round's final eval.
        initial_frame = None

        while True:
            round_num += 1
            if round_num > 1:
                logger.info("=" * 60)
                logger.info(f"ROUND {round_num} for '{prompt}' — re-decomposing "
                            f"against the current scene ({retries_left} retry "
                            f"round(s) left after this one).")
                logger.info("=" * 60)

            result = _run_high_level_task_body(
                client, planner, frames, cfg, prompt, interjection, logger,
                round_num=round_num, retry_context=retry_context,
                retry_objects=retry_objects, initial_frame=initial_frame,
            )
            if initial_frame is None:
                initial_frame = result.initial_frame

            if result.stopped:
                return
            if result.final_success is None:
                # No verdict to act on (final eval disabled, unparseable, or it
                # errored) — the round's own logging already said why.
                return
            if result.final_success:
                logger.info(f"High-level task '{prompt}' judged COMPLETE "
                            f"after {round_num} round(s).")
                return
            if retries_left <= 0:
                logger.warning(
                    f"High-level task '{prompt}' judged INCOMPLETE after "
                    f"{round_num} round(s) and no retry rounds remain: "
                    f"{result.final_reason}"
                )
                return
            if round_num > 1 and result.planned == 0:
                # The re-decompose saw the completed list and found nothing left
                # to do, so the final evaluator and the planner disagree. Another
                # round would plan nothing either (the scene cannot change), so
                # stop rather than burn the remaining budget on empty rounds.
                logger.info(
                    f"High-level task '{prompt}' judged INCOMPLETE, but the "
                    f"re-decomposition found nothing left to plan against the "
                    f"current scene — stopping. Final evaluator said: "
                    f"{result.final_reason}"
                )
                return

            for st in result.completed:
                if st not in completed_all:
                    completed_all.append(st)

            retries_left -= 1
            reason = result.final_reason or "(no reason given)"
            retry_context = RETRY_DECOMPOSE_CONTEXT.format(
                reason=reason,
                completed=_format_completed_context(completed_all),
            )
            retry_objects = None
            if cfg.targeted_retry_decompose and result.final_survey:
                remaining, destinations = remaining_from_survey(result.final_survey)
                if not remaining:
                    # The judge says incomplete, but its own survey puts every
                    # object on a destination: there is nothing to re-plan, and a
                    # forced plan would only invent work (arm, table, tray).
                    logger.info(
                        f"High-level task '{prompt}' judged INCOMPLETE, but the "
                        f"final-eval survey finds no object off a destination "
                        f"({destinations}) — nothing to re-plan, stopping. Final "
                        f"evaluator said: {result.final_reason}"
                    )
                    return
                retry_objects = remaining + destinations + ["table"]
                retry_context += "\n\n" + TARGETED_RETRY_NOTE
                logger.info(f"Targeted retry: survey found {remaining} not at a "
                            f"destination; planning over {retry_objects}.")
            logger.info(f"Carrying {len(completed_all)} completed sub-task(s) into "
                        f"the re-decompose: {completed_all}")
            logger.info(f"Retry context passed to the planner:\n{retry_context}")
            logger.warning(
                f"Final evaluation judged '{prompt}' INCOMPLETE: "
                f"{result.final_reason} — running the whole loop again."
            )
    finally:
        log_sink.close()


# ---------------------------------------------------------------------------
# Vocabulary refinement helpers
# ---------------------------------------------------------------------------

_INSTRUCTION_PATTERN = re.compile(
    r"^(put\s+the\s+)(.+?)(\s+(?:on|in|into|onto|next\s+to|near)\s+.+)$",
    re.IGNORECASE,
)


def extract_object_name(instruction: str) -> str:
    """Pull the object noun from 'put the X on the tray'."""
    m = _INSTRUCTION_PATTERN.match(instruction.strip())
    return m.group(2) if m else instruction


def rebuild_instruction(original: str, new_label: str) -> str:
    """Replace the object name in a pick-and-place instruction."""
    m = _INSTRUCTION_PATTERN.match(original.strip())
    if m:
        return f"{m.group(1)}{new_label}{m.group(3)}"
    return original


def log_attempt(logger, *, attempt, original_instruction, current_instruction,
                success, failure_type, reason, refinement_applied):
    logger.info(
        f"ATTEMPT attempt={attempt} "
        f"original={original_instruction!r} "
        f"current={current_instruction!r} "
        f"success={success} "
        f"failure_type={failure_type} "
        f"reason={reason!r} "
        f"refinement={refinement_applied}"
    )


# ---------------------------------------------------------------------------


def _run_high_level_task_body(
    client: LoopRobotClient,
    planner: VLMPlanner,
    frames: VLMFrameSource,
    cfg: OrchestratorConfig,
    prompt: str,
    interjection: InterjectionManager,
    logger,
    round_num: int = 1,
    retry_context: str = "",
    retry_objects: list[str] | None = None,
    initial_frame: "Image.Image | None" = None,
) -> RoundResult:
    def abort(where: str):
        """Park the arm and report that the operator abandoned the task."""
        logger.warning(f"ABORT requested {where} — abandoning high-level task "
                       f"'{prompt}' and returning to the prompt.")
        client.go_home()

    # 1. Decompose with a fresh observation. On a retry round `retry_context`
    #    carries the final evaluator's verdict plus every sub-task earlier rounds
    #    completed, so the planner sees what is already done and what the previous
    #    round left undone as well as the current scene.
    tag = "decompose" if round_num == 1 else f"decompose_round{round_num}"
    logger.info(f"Decomposing high-level prompt (round {round_num}): '{prompt}'")
    frame = frames.capture(tag=tag)
    if initial_frame is None:
        initial_frame = frame  # round 1: the scene before the robot started
    try:
        subtasks = planner.decompose(prompt, frame, extra_context=retry_context,
                                     objects=retry_objects)
    except ValueError as e:
        logger.error(str(e))
        logger.error("Decomposition failed — skipping this prompt.")
        return RoundResult(stopped=True)

    # Abort typed while the VLM was decomposing: nothing has moved yet.
    itype, _ = interjection.check_and_consume()
    if itype == InterjectionType.ABORT:
        abort("during decomposition")
        return RoundResult(stopped=True)
    elif itype != InterjectionType.NONE:
        logger.info(f"Ignoring {itype.value} requested during decomposition "
                    f"— no sub-task has run yet.")

    logger.info(f"Sub-task queue ({len(subtasks)}):")
    for i, st in enumerate(subtasks, 1):
        logger.info(f"  {i}. {st}")

    # Track progress for replanning context
    completed: list[str] = []
    replans_remaining = max(cfg.max_replans, 0)

    # Use a mutable list so replan can replace the remaining queue
    pending = list(subtasks)

    def do_replan(failed_subtask: str, failure_reason: str,
                  tag: str) -> list[str] | None:
        """Capture a frame and replan; returns the new queue, or None on failure."""
        replan_frame = frames.capture(tag=tag)
        try:
            new_queue = planner.replan(
                original_prompt=prompt,
                completed_subtasks=completed,
                failed_subtask=failed_subtask,
                failure_reason=failure_reason,
                frame=replan_frame,
            )
        except ValueError as e:
            logger.error(f"Replan failed ({e}) — continuing with remaining queue.")
            return None
        logger.info(f"Replanned queue ({len(new_queue)}):")
        for i, st in enumerate(new_queue, 1):
            logger.info(f"  {i}. {st}")
        return new_queue

    # The VLM monitor (reused across sub-tasks); it shares the planner's model.
    monitor = (LiveMonitor(client, planner.model, planner.processor, cfg,
                           temperature=planner.temperature,
                           freeze_dir=frames.task_dir / "monitor_freeze")
               if cfg.enable_vlm_monitor else None)

    # 2. Work through the sub-task queue
    episode_count = 0  # names this round's per-episode monitor records uniquely
    while pending:
        sub_task = pending.pop(0)
        task_num = len(completed) + 1
        attempts_left = 1 + max(cfg.max_retries, 0)
        succeeded = False
        no_movement = False
        user_replanned = False
        user_skipped = False
        user_aborted = False
        reason = ""  # last failure reason, passed to replan
        current_instruction = sub_task
        vocab_refinement_used = False
        monitor_stops = 0         # VLM monitor STOPs on this sub-task
        monitor_skipped = False   # second STOP: the sub-task is skipped, no replan

        while attempts_left > 0:
            attempts_left -= 1
            # Reflects this attempt only: a skip on attempt N followed by a real
            # failure on attempt N+1 must still replan like any other failure.
            user_skipped = False

            logger.info(f"[{task_num}] Executing sub-task: '{current_instruction}' "
                        f"({attempts_left} retries left)"
                        + (f" [original: '{sub_task}']"
                           if current_instruction != sub_task else ""))
            episode_count += 1
            attempt_tag = (f"round{round_num}_ep{episode_count:02d}_task{task_num:02d}"
                           f"_attempt{(1 + max(cfg.max_retries, 0)) - attempts_left}")
            if monitor is not None:
                monitor.freeze_dir = frames.task_dir / "monitor_freeze"
                monitor.start(current_instruction, tag=attempt_tag)
            ep_result = client.run_episode(current_instruction,
                                           enable_abort_listener=False)
            monitor_result = None
            if monitor is not None:
                monitor.stop()
                summary = monitor.summary()
                try:
                    (frames.task_dir / f"monitor_{attempt_tag}.json").write_text(
                        json.dumps(summary, indent=2, default=str))
                except OSError as e:
                    logger.warning(f"Could not save the monitor record: {e}")
                if monitor.stopped:
                    # The reach check's reason plus what the wrist probe saw at the grasp.
                    monitor_result = {"reason": f"{monitor.stopped['reason']} (wrist check: "
                                                f"{monitor.stopped['verify']})"}
                logger.info(f"VLM monitor: {summary['outcome']}"
                            + (f" — committed to '{monitor.held_object}'"
                               if monitor.committed else "")
                            + f" | {monitor.reach_windows} reach check(s), "
                              f"{monitor.verify_calls} verify call(s)")

            # Open the gripper (an object held when the episode ended is let go here,
            # not dragged home), lift away, and return home so the arm is out of frame
            # and the next episode starts from a consistent pose.
            client.release_and_home()

            # --- Interjection check: after episode ---
            itype, _ = interjection.check_and_consume()
            if itype == InterjectionType.ABORT:
                abort(f"during sub-task '{sub_task}'")
                user_aborted = True
                break
            elif itype == InterjectionType.SKIP:
                user_skipped = True
                reason = "The user skipped this episode."
                logger.info(f"User skipped sub-task: '{sub_task}' "
                            f"({attempts_left} attempt(s) left)")
                continue
            elif itype == InterjectionType.REPLAN:
                logger.info("User requested replan — stopping and replanning...")
                new_queue = do_replan(sub_task, "The user requested a replan.",
                                      "user_replan")
                if new_queue is not None:
                    pending = new_queue
                user_replanned = True
                break

            # --- Interjection check: after the release and homing ---
            itype, _ = interjection.check_and_consume()
            if itype == InterjectionType.ABORT:
                abort(f"during sub-task '{sub_task}'")
                user_aborted = True
                break
            elif itype == InterjectionType.SKIP:
                user_skipped = True
                reason = "The user skipped this episode."
                logger.info(f"User skipped sub-task: '{sub_task}' "
                            f"({attempts_left} attempt(s) left)")
                continue
            elif itype == InterjectionType.REPLAN:
                logger.info("User requested replan — stopping and replanning...")
                new_queue = do_replan(sub_task, "The user requested a replan.",
                                      "user_replan")
                if new_queue is not None:
                    pending = new_queue
                user_replanned = True
                break

            # --- No-movement detection ---
            no_movement = (ep_result.converged
                           and ep_result.max_displacement < cfg.no_movement_threshold)

            if no_movement:
                logger.warning(
                    f"No movement detected (max_displacement="
                    f"{ep_result.max_displacement:.4f}). The object name "
                    f"'{extract_object_name(current_instruction)}' is likely "
                    f"not in the robot's vocabulary."
                )

            # --- VLM monitor stopped the episode ---
            # A STOP means the monitor caught the arm going for the wrong object at the
            # grasp, so the attempt is a known failure and is not sent to the evaluator.
            # The first STOP retries the same instruction once (even when the normal
            # retries are used up); a second STOP on the sub-task skips it, with no
            # label refinement or replan.
            if monitor_result is not None:
                monitor_stops += 1
                reason = monitor_result["reason"]
                logger.info(f"VLM monitor stopped the episode ({monitor_stops}): {reason}")
                log_attempt(
                    logger,
                    attempt=(1 + max(cfg.max_retries, 0)) - attempts_left,
                    original_instruction=sub_task,
                    current_instruction=current_instruction,
                    success=False,
                    failure_type="MONITOR_STOP",
                    reason=reason,
                    refinement_applied=(current_instruction != sub_task),
                )
                if monitor_stops == 1:
                    attempts_left = max(attempts_left, 1)
                    logger.info(f"Monitor STOP — retrying '{current_instruction}' once.")
                    continue
                logger.warning(f"Second monitor STOP on '{sub_task}' — skipping it.")
                monitor_skipped = True
                break

            if not cfg.evaluate_subtasks:
                succeeded = True
                break
            else:
                # 3. Judge success — from a video clip of the attempt if available,
                #    otherwise a fresh still frame.
                clip_fps = None
                if cfg.vlm_eval_use_video:
                    observation, clip_span = frames.capture_clip(
                        tag=f"eval_task{task_num}",
                        num_frames=cfg.vlm_eval_num_frames,
                    )
                    if not observation:
                        observation = frames.capture(tag=f"eval_task{task_num}")
                        logger.info("Evaluation observation: static image "
                                    "(video requested but clip buffer was empty).")
                    else:
                        if clip_span > 0 and len(observation) > 1:
                            clip_fps = (len(observation) - 1) / clip_span
                        logger.info(
                            f"Evaluation observation: video clip "
                            f"({len(observation)} frames over {clip_span:.1f}s "
                            f"of a {ep_result.duration:.1f}s episode, "
                            + (f"{clip_fps:.2f} fps)" if clip_fps
                               else f"nominal {cfg.vlm_eval_fps} fps)")
                        )
                else:
                    observation = frames.capture(tag=f"eval_task{task_num}")
                    logger.info("Evaluation observation: static image "
                                "(vlm_eval_use_video disabled).")
                eval_result = None
                if (cfg.subtask_done_precheck and no_movement
                        and isinstance(observation, list) and len(observation) >= 2):
                    # The arm did not move: is the object already on the tray? Then
                    # the attempt is done (the evaluator would call it "no attempt").
                    try:
                        pre = planner.precheck_subtask_done(
                            current_instruction, observation[0], observation[-1])
                        logger.info(f"Sub-task pre-check: done={pre['done']} | {pre['reason']}")
                        if pre["done"]:
                            eval_result = {"success": True,
                                           "reason": f"Pre-check: {pre['reason']}."}
                    except Exception as e:
                        logger.warning(f"Sub-task pre-check failed ({e}); running the "
                                       "evaluator.")
                if eval_result is None:
                    try:
                        eval_result = planner.evaluate(current_instruction, observation,
                                                       fps=clip_fps)
                    except ValueError as e:
                        logger.warning(f"Could not parse VLM evaluation ({e}); "
                                       "assuming success and moving on.")
                        succeeded = True
                        break
                    except Exception as e:
                        logger.exception(
                            f"VLM evaluation errored ({e}); abandoning high-level task "
                            f"'{prompt}' — sub-task success can no longer be verified. "
                            f"Completed and verified before the failure: {completed}"
                        )
                        return RoundResult(stopped=True, planned=len(subtasks),
                                           completed=list(completed))

                itype, _ = interjection.check_and_consume()
                if itype == InterjectionType.ABORT:
                    abort("during evaluation")
                    user_aborted = True
                    break
                elif itype != InterjectionType.NONE:
                    logger.info(f"Ignoring {itype.value} requested during evaluation "
                                f"— '{current_instruction}' was already judged.")

                success = bool(eval_result.get("success"))
                reason = eval_result.get("reason", "")
                failure_type = eval_result.get("failure_type", "RETRY")
                if no_movement and not success:
                    failure_type = "VOCAB"
                logger.info(f"VLM judgement: success={success} | "
                            f"failure_type={failure_type} | {reason}")

            if success:
                succeeded = True
                break

            # --- Structured attempt logging ---
            attempt_num = (1 + max(cfg.max_retries, 0)) - attempts_left
            log_attempt(
                logger,
                attempt=attempt_num,
                original_instruction=sub_task,
                current_instruction=current_instruction,
                success=False,
                failure_type=failure_type,
                reason=reason,
                refinement_applied=(current_instruction != sub_task),
            )

            # --- Failure-classified retry logic ---
            if (cfg.enable_vocab_refinement
                    and failure_type == "VOCAB"
                    and attempts_left > 0
                    and not vocab_refinement_used):
                logger.info("VOCAB failure — attempting label refinement...")
                eval_frame = frames.capture(tag=f"refine_task{task_num}")
                try:
                    alternatives = planner.refine_label(
                        current_instruction, reason, frame=eval_frame)
                except Exception as e:
                    logger.warning(f"Refinement call failed ({e}); "
                                   "retrying with current instruction.")
                    alternatives = []
                if alternatives:
                    new_instr = rebuild_instruction(
                        current_instruction, alternatives[0])
                    logger.info(f"Refined: '{current_instruction}' -> "
                                f"'{new_instr}' "
                                f"(candidates: {alternatives})")
                    current_instruction = new_instr
                vocab_refinement_used = True
            elif attempts_left > 0:
                logger.info(f"Retrying sub-task ({attempts_left} attempt(s) "
                            f"left)...")

        if user_aborted:
            logger.info(f"  Completed before the abort: {completed}")
            return RoundResult(stopped=True, planned=len(subtasks),
                               completed=list(completed))

        if user_replanned:
            continue

        if succeeded:
            completed.append(sub_task)
            continue

        # --- Monitor STOPped the sub-task twice ---
        # Counts as failed (never added to `completed`); no replan.
        if monitor_skipped:
            logger.warning(f"Sub-task '{sub_task}' skipped after two monitor STOPs — "
                           f"counting as failed and moving on.")
            continue

        # --- User skipped the final attempt ---
        # Counts as failed (never added to `completed`) but does not trigger a
        # replan: the user asked to move on, not to revise the plan.
        if user_skipped:
            logger.warning(f"Sub-task '{sub_task}' skipped by user on the final "
                           f"attempt — counting as failed and moving on.")
            continue

        # --- All retries exhausted for this sub-task ---
        logger.warning(f"Sub-task '{sub_task}' failed after all retries.")

        if replans_remaining <= 0:
            logger.warning("No replans remaining — skipping failed sub-task "
                           "and continuing with the rest of the queue.")
            continue

        # 4. Replan the remaining sequence
        replans_remaining -= 1
        logger.info(f"Triggering replan ({replans_remaining} replan(s) left after this)...")
        logger.info(f"  Completed so far: {completed}")
        logger.info(f"  Failed: '{sub_task}' — {reason}")
        logger.info(f"  Remaining (discarded): {pending}")

        new_subtasks = do_replan(sub_task, reason, "replan")
        if new_subtasks is None:
            continue

        # Replace the pending queue with the replanned tasks
        pending = new_subtasks

    # 5. Final whole-task evaluation against the original prompt. Reached only
    #    when the queue empties normally — every abort / early return above skips
    #    this. A failed verdict is returned to the caller, which decomposes the
    #    prompt again and runs another round (up to --max_final_retries).
    result = RoundResult(planned=len(subtasks), completed=list(completed),
                         initial_frame=initial_frame)
    if cfg.evaluate_final:
        logger.info("Running final evaluation against the original prompt...")
        final_frame = frames.capture(
            tag="final_eval" if round_num == 1 else f"final_eval_round{round_num}"
        )
        try:
            final = planner.evaluate_final(
                prompt, final_frame,
                initial_frame if cfg.final_eval_before_after else None)
            result.final_success = bool(final.get("success"))
            result.final_survey = final.get("surveyed") or []
            result.final_reason = final.get("reason", "")
            logger.info(f"FINAL VLM judgement for '{prompt}' (round {round_num}): "
                        f"success={result.final_success} | {result.final_reason}")
        except ValueError as e:
            logger.warning(f"Could not parse final evaluation ({e}); no overall verdict.")
        except Exception as e:
            logger.exception(f"Final evaluation errored ({e}); no overall verdict.")

    # An abort typed while the final evaluation was running ends the whole task:
    # it must not be carried silently into a retry round.
    if interjection.check_and_consume()[0] == InterjectionType.ABORT:
        abort("during final evaluation")
        result.stopped = True

    logger.info(f"Round {round_num} complete for: '{prompt}'")
    logger.info(f"  Completed {len(completed)}/{len(subtasks)} sub-tasks "
                f"planned this round: {completed}")
    return result


# ---------------------------------------------------------------------------
# Terminal capture
# ---------------------------------------------------------------------------
class TaskLogSink:
    """A swappable mirror target for the terminal tees.

    Holds the currently-active per-high-level-task log file. `open()` points it
    at a new file (closing any previous one); `close()` detaches it. Between
    tasks `fh` is None, so nothing is mirrored. Shared by both stdout/stderr
    tees so a single `open()` captures both streams into one file.
    """

    def __init__(self):
        self.fh = None

    def open(self, path):
        self.close()
        self.fh = open(path, "a", buffering=1, encoding="utf-8")  # line-buffered

    def write(self, data):
        if self.fh is not None:
            try:
                self.fh.write(data)
            except (ValueError, OSError):
                pass

    def flush(self):
        if self.fh is not None:
            try:
                self.fh.flush()
            except (ValueError, OSError):
                pass

    def close(self):
        self.flush()
        if self.fh is not None:
            try:
                self.fh.close()
            except (ValueError, OSError):
                pass
        self.fh = None


class _Tee:
    """Duplicate a text stream to a swappable sink (the active per-task log)."""

    def __init__(self, primary, sink: "TaskLogSink"):
        self._primary = primary   # original sys.stdout / sys.stderr
        self._sink = sink         # shared TaskLogSink (per-task mirror)

    def write(self, data):
        self._primary.write(data)
        self._sink.write(data)
        return len(data)

    def flush(self):
        self._primary.flush()
        self._sink.flush()

    def isatty(self):
        return self._primary.isatty()

    def fileno(self):
        return self._primary.fileno()

    def __getattr__(self, name):
        return getattr(self._primary, name)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
@draccus.wrap()
def main(cfg: OrchestratorConfig):
    logging.basicConfig(level=logging.INFO)

    # Single per-run root holding one sub-folder per high-level task; each
    # sub-folder gets that task's log (task.log) and, when --save_frames is on,
    # its saved frames. One timestamp for the whole run.
    run_dir = Path(f"./runs/run_{datetime.now():%Y-%m-%d_%H-%M-%S}")
    run_dir.mkdir(parents=True, exist_ok=True)
    # Keep the action log inside the run too; run_high_level_task then repoints
    # it at the active task's sub-folder, so this file only ever holds actions
    # sent outside a task (e.g. none in practice).
    cfg.action_log_dir = str(run_dir)

    # Mirror every byte written to the terminal (raw print()/input() as well as
    # logging output and third-party library prints) into the currently-active
    # per-task log, so each high-level task can be reviewed on its own. The sink
    # is swapped at each task boundary (see run_high_level_task); between tasks
    # nothing is mirrored.
    log_sink = TaskLogSink()
    _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(_orig_stdout, log_sink)
    sys.stderr = _Tee(_orig_stderr, log_sink)

    # The root logger's console handler grabbed the ORIGINAL sys.stderr at import
    # time, so re-point it at the tee'd stderr; leave file handlers untouched.
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setStream(sys.stderr)

    logging.info(f"Per-task logs will be written under {run_dir}")
    logging.info(pformat(asdict(cfg)))

    # 1. Robot + policy server (VLA layer)
    # TappedClient: the base client plus the monitor's frame tap, per-tick grasp
    # watch, pause throttling and the end-of-episode release.
    if cfg.enable_vlm_monitor:
        cfg.log_vla_inputs = True  # the monitor's frame tap relies on the input capture
        if cfg.wrist_key not in (getattr(cfg.robot, "cameras", None) or {}):
            logging.warning(f"VLM monitor: no '{cfg.wrist_key}' camera in --robot.cameras — "
                            "the wrist verify will be blind and its STOPs will stand.")
    client = TappedClient(cfg)
    if not client.connect_server():
        client.logger.error("Could not connect to policy server. Exiting.")
        client.stop()
        return
    # 2. VLM planner (reasoning layer)
    planner = VLMPlanner(cfg.vlm_model, temperature=cfg.vlm_temperature,
                         eval_fps=cfg.vlm_eval_fps,
                         two_pass=cfg.vlm_two_pass_decompose,
                         reask_uncovered=cfg.vlm_reask_uncovered,
                         filter_fixtures=cfg.vlm_filter_fixtures)
    # 3. Frame source for the VLM. session_dir is the per-run root (shared with
    #    the task logs); each high-level task gets its own sub-folder underneath
    #    (see start_task). Frame PNGs are saved only when --save_frames is on.
    frames = VLMFrameSource(client, camera_key=cfg.vlm_camera_key,
                            session_dir=run_dir, save_frames=cfg.save_frames)
    # 4. Interjection manager (user can skip/replan mid-execution)
    interjection = InterjectionManager(client)

    client.logger.info("=" * 60)
    client.logger.info("Dual-system ready: VLM planner + VLA policy loaded.")
    client.logger.info(f"Episode duration per sub-task: {cfg.episode_duration}s")
    client.logger.info(f"Max retries per sub-task: {cfg.max_retries}")
    client.logger.info(f"Max replans per prompt: {cfg.max_replans}")
    if cfg.evaluate_final:
        client.logger.info(
            f"Final eval: on — up to {cfg.max_final_retries} extra full round(s) "
            f"if the whole task is judged incomplete."
        )
    elif cfg.max_final_retries:
        client.logger.warning("--max_final_retries is set but --evaluate_final is "
                              "off; no final verdict, so no retry rounds will run.")
    client.logger.info("Replans pause for operator context — type guidance and press "
                       "Enter, or Enter alone to skip.")
    client.logger.info("Type an instruction and press Enter to execute.")
    if cfg.enable_interjection:
        client.logger.info("During execution: 's'+Enter to skip, 'r'+Enter to replan.")
    client.logger.info(f"At any point during execution (and at any replan prompt): "
                       f"'{ABORT_KEY}'+Enter aborts the whole task and returns here.")
    client.logger.info("Type 'quit' or 'exit' to shut down.")
    client.logger.info("=" * 60)

    try:
        while True:
            try:
                prompt = input("\n[READY] Enter high-level prompt: ").strip()
            except EOFError:
                break

            if not prompt:
                continue
            if prompt.lower() in ("quit", "exit"):
                client.logger.info("Shutdown requested.")
                break

            # The listener always runs so 'q'+Enter can abort at any time; with
            # --enable_interjection=false it honours nothing else.
            interjection.start(abort_only=not cfg.enable_interjection)
            try:
                run_high_level_task(client, planner, frames, cfg, prompt,
                                    interjection, log_sink)
            finally:
                interjection.stop()

    except KeyboardInterrupt:
        client.logger.info("\nInterrupted by user.")

    finally:
        client.go_home()
        frames.stop()
        client.stop()
        sys.stdout, sys.stderr = _orig_stdout, _orig_stderr
        # Close any task log still open (e.g. interrupted mid-task).
        log_sink.close()


if __name__ == "__main__":
    register_third_party_plugins()
    main()