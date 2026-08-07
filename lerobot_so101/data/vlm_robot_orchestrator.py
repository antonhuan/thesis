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
#        |
#        v
#   [VLM: success evaluation]  -- retry on failure, replan if retries exhausted
#
# Usage:
#   Terminal 1 (policy server, stays running):
#     python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080
#
#   Terminal 2 (this script):
#     python vlm_robot_orchestrator.py \
#       --robot.type=so101_follower \
#       --robot.port=/dev/ttyACM0 \
#       --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }" \
#       --task="idle" \
#       --server_address=127.0.0.1:8080 \
#       --policy_type=smolvla \
#       --pretrained_name_or_path=edge-inference/smolvla-so101-pick-orange \
#       --policy_device=cuda \
#       --actions_per_chunk=50 \
#       --chunk_size_threshold=0.7 \
#       --episode_duration=30 \
#       --vlm_model=Qwen/Qwen3-VL-4B-Instruct \
#       --evaluate_subtasks=true \
#       --vlm_eval_use_video=true \
#       --vlm_eval_num_frames=16 \
#       --clip_buffer_maxlen=128 \
#       --max_retries=3 \
#       --max_replans=1 \
#       --enable_interjection=true
#
# At the prompt, type a HIGH-LEVEL instruction (e.g. "clean up the table but
# leave the cups"). The VLM decomposes it into sub-tasks, each sub-task is
# executed by the VLA policy, and (optionally) the VLM judges success from a
# fresh camera frame, retrying failed sub-tasks. If retries are exhausted the
# VLM replans the remaining sequence given the current scene state.
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
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import numpy as np
from PIL import Image

from lerobot.utils.import_utils import register_third_party_plugins

import draccus

from robot_client_loop import LoopClientConfig, LoopRobotClient
from vlm_core import generate, load_model
from vlm import (
    DECOMPOSITION_SYSTEM_PROMPT,
    identify_objects,
    decompose_from_objects,
)
from datetime import datetime

# ---------------------------------------------------------------------------
# Config: extends the loop client config with VLM parameters
# ---------------------------------------------------------------------------
@dataclass
class OrchestratorConfig(LoopClientConfig):
    # HuggingFace model name or local path for the VLM planner
    vlm_model: str = "Qwen/Qwen3-VL-4B-Instruct"
    # Sampling temperature for VLM generation
    vlm_temperature: float = 0.1
    # Split decomposition into two model calls against the same frame:
    # (1) identify visible objects, then (2) decompose given that object list.
    # Set false to use the single-call DECOMPOSITION_SYSTEM_PROMPT instead.
    vlm_two_pass_decompose: bool = True
    # Use the dedicated top-down RealSense D435 for VLM input.
    # If false (or the camera fails to start), falls back to the robot's own
    # camera observation, then to text-only.
    vlm_camera_key: str = "front"
    # After each sub-task episode, ask the VLM whether it succeeded
    evaluate_subtasks: bool = True
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
    # mid-execution (Enter/'s' to skip, 'r'+Enter to replan). On by default so
    # skip works out of the box; pass --enable_interjection=false to disable.
    enable_interjection: bool = False
    # Max joint displacement (L2) below which a converged episode counts as
    # "no movement" — i.e. the VLA did not understand the instruction.
    # Tunable: start at 0.02 and adjust based on your arm's joint scale.
    no_movement_threshold: float = 0.02


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
- Judge ONLY whether the specific object named in the sub-task is now at the destination. Do not assess other objects.

Output format:
Return ONLY a JSON object with two fields:
- "success": true or false
- "reason": a brief explanation of your judgement

Example:
{"success": true, "reason": "The apple is now on the tray as instructed."}
{"success": false, "reason": "The banana is still on the table, not on the tray."}
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

    def prompt_for_context(self, header: str) -> str:
        """Block on stdin for one line of operator guidance and return it.

        Typing the abort key instead records an ABORT (retrievable via
        check_and_consume) and returns "".

        Pauses the background listener first so it does not swallow the typed
        line. Safe to call when no listener is running: it falls back to a
        plain input().
        """
        listener_running = self._thread is not None and self._active.is_set()
        hint = (f"[REPLAN] Additional context for the planner "
                f"(Enter for none, '{ABORT_KEY}' to abort): ")

        if not listener_running:
            print(f"\n{header}")
            try:
                line = input(hint).strip()
            except EOFError:
                return ""
            return self._consume_context_line(line)

        self._paused.set()
        # Longer than the listener's select() timeout, so any in-flight poll
        # returns and observes _paused before touching stdin.
        time.sleep(0.25)
        try:
            print(f"\n{header}")
            print(hint, end="", flush=True)
            line = sys.stdin.readline()
            return self._consume_context_line(line.strip() if line else "")
        finally:
            self._paused.clear()

    def _consume_context_line(self, line: str) -> str:
        if line.lower() == ABORT_KEY:
            self.request_abort()
            return ""
        return line

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
                    # End the episode immediately; context is collected on the
                    # main thread once the arm has stopped (prompt_for_context),
                    # so the robot never keeps executing while the user types and
                    # the episode can't naturally end mid-typing and drop the replan.
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


def parse_subtask_list(output: str) -> list[str]:
    text = _strip_code_fences(output.strip())

    try:
        result = json.loads(text)
        if isinstance(result, dict) and "subtasks" in result:
            tasks = result["subtasks"]
            print(f"\nVisible: {result.get('visible_objects')}")
            print(f"Excluded: {result.get('excluded_objects')}")
            print(f"Allowed: {result.get('allowed_objects')}")
        else:
            tasks = result
        print(f"\nParsed {len(tasks)} sub-tasks:")
        for i, task in enumerate(tasks, 1):
            print(f"  {i}. {task}")
        return tasks
    except json.JSONDecodeError:
        raise ValueError(f"Could not parse subtasks from VLM output:\n{output}")

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
                 eval_fps: float = 2.0, two_pass: bool = True):
        self.model, self.processor = load_model(model_name)
        self.temperature = temperature
        self.eval_fps = eval_fps
        # Two-pass decomposition: identify objects, then decompose given the list.
        self.two_pass = two_pass

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

    def decompose(self, prompt: str, frame: Image.Image | None) -> list[str]:
        """High-level prompt + observation -> ordered list of sub-tasks.

        Two-pass (default) identifies the visible objects first, then decomposes
        given that list; both calls share the single `frame`. Falls back to the
        single-call prompt when two-pass is disabled or there is no frame to
        identify objects from.
        """
        if self.two_pass and frame is not None:
            image_content = [
                {"type": "text", "text": "[top-down camera]"},
                {"type": "image", "image": frame},
            ]
            visible = identify_objects(
                self.model, self.processor, image_content, self.temperature
            )
            logging.info(f"Identified objects ({len(visible)}): {visible}")
            output = decompose_from_objects(
                self.model, self.processor, image_content, prompt, visible,
                self.temperature,
            )
        else:
            text = f"\nInstruction: {prompt}"
            if frame is None:
                text = (
                    "No camera observation is available. Decompose the instruction "
                    "based on the text alone.\n" + text
                )
            messages = [
                {"role": "system", "content": [{"type": "text", "text": DECOMPOSITION_SYSTEM_PROMPT}]},
                {"role": "user", "content": self._user_content(frame, text)},
            ]
            output = generate(
                self.model, self.processor, messages, temperature=self.temperature
            )
        logging.info(f"Raw VLM decomposition output ({len(output)} chars): {output!r}")
        return parse_subtask_list(output)

    def evaluate(
        self,
        sub_task: str,
        observation: "Image.Image | list[Image.Image] | None",
        fps: float | None = None,
    ) -> dict:
        """Observation + attempted sub-task -> {'success': bool, 'reason': str}.

        `observation` may be a single PIL frame (still) or a list of PIL frames
        (a video clip of the attempt). An empty list is treated as no observation.

        `fps` overrides the configured nominal rate — pass the clip's true rate
        (frames / episode duration) so the model's frame timestamps span the
        real length of the attempt.
        """
        text = (
            f'\nThe robot just attempted this sub-task: "{sub_task}"\nDid it succeed?'
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
        logging.info(f"Raw eval output: {output}")
        return parse_evaluation(output)

    def replan(
        self,
        original_prompt: str,
        completed_subtasks: list[str],
        failed_subtask: str,
        failure_reason: str,
        frame: Image.Image | None,
        user_context: str = "",
    ) -> list[str]:
        """Replan remaining sub-tasks given what succeeded, what failed, the
        current scene state, and any guidance typed by the human operator."""
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

        if user_context:
            context_parts.append(
                "Additional guidance from the human operator (authoritative — "
                "follow it over your own reading of the scene):\n"
                f"{user_context}"
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
        logging.info(f"Raw VLM replan output ({len(output)} chars): {output!r}")
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
            for i, img in enumerate(clip):
                path = self.task_dir / f"{tag}_{ts}_{i:02d}.png"
                img.save(path)
            logging.info(f"Saved VLM eval clip ({len(clip)} frames) to {self.task_dir}")

        return clip, span

    def stop(self):
        pass  # No separate pipeline to clean up


# ---------------------------------------------------------------------------
# Orchestration: one high-level prompt -> decompose -> execute -> evaluate
# ---------------------------------------------------------------------------
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
    task_dir = frames.start_task()
    log_sink.open(task_dir / "task.log")
    try:
        _run_high_level_task_body(
            client, planner, frames, cfg, prompt, interjection, logger
        )
    finally:
        log_sink.close()


def _run_high_level_task_body(
    client: LoopRobotClient,
    planner: VLMPlanner,
    frames: VLMFrameSource,
    cfg: OrchestratorConfig,
    prompt: str,
    interjection: InterjectionManager,
    logger,
):
    def abort(where: str):
        """Park the arm and report that the operator abandoned the task."""
        logger.warning(f"ABORT requested {where} — abandoning high-level task "
                       f"'{prompt}' and returning to the prompt.")
        client.go_home()

    # 1. Decompose with a fresh observation
    logger.info(f"Decomposing high-level prompt: '{prompt}'")
    frame = frames.capture(tag="decompose")
    try:
        subtasks = planner.decompose(prompt, frame)
    except ValueError as e:
        logger.error(str(e))
        logger.error("Decomposition failed — skipping this prompt.")
        return

    # Abort typed while the VLM was decomposing: nothing has moved yet.
    itype, _ = interjection.check_and_consume()
    if itype == InterjectionType.ABORT:
        abort("during decomposition")
        return
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
                  user_context: str, tag: str) -> list[str] | None:
        """Capture a frame and replan; returns the new queue, or None on failure."""
        replan_frame = frames.capture(tag=tag)
        try:
            new_queue = planner.replan(
                original_prompt=prompt,
                completed_subtasks=completed,
                failed_subtask=failed_subtask,
                failure_reason=failure_reason,
                frame=replan_frame,
                user_context=user_context,
            )
        except ValueError as e:
            logger.error(f"Replan failed ({e}) — continuing with remaining queue.")
            return None
        logger.info(f"Replanned queue ({len(new_queue)}):")
        for i, st in enumerate(new_queue, 1):
            logger.info(f"  {i}. {st}")
        return new_queue

    # 2. Work through the sub-task queue
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

        while attempts_left > 0:
            attempts_left -= 1
            # Reflects this attempt only: a skip on attempt N followed by a real
            # failure on attempt N+1 must still replan like any other failure.
            user_skipped = False

            logger.info(f"[{task_num}] Executing sub-task: '{sub_task}' "
                        f"({attempts_left} retries left)")
            ep_result = client.run_episode(sub_task, enable_abort_listener=False)

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
                client.go_home()
                continue
            elif itype == InterjectionType.REPLAN:
                logger.info("User requested replan — stopping and asking for context...")
                client.go_home()
                user_context = interjection.prompt_for_context(
                    f"[REPLAN] Replan requested during sub-task '{sub_task}'"
                )
                # The operator can type the abort key instead of context.
                if interjection.check_and_consume()[0] == InterjectionType.ABORT:
                    abort("at the replan prompt")
                    user_aborted = True
                    break
                if user_context:
                    logger.info(f"  Operator context: {user_context}")
                new_queue = do_replan(sub_task, "The user requested a replan.",
                                      user_context, "user_replan")
                if new_queue is not None:
                    pending = new_queue
                user_replanned = True
                break

            # Return home so the arm is out of frame and the next episode
            # starts from a consistent pose
            client.go_home()

            # --- Interjection check: after go_home ---
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
                logger.info("User requested replan — stopping and asking for context...")
                user_context = interjection.prompt_for_context(
                    f"[REPLAN] Replan requested during sub-task '{sub_task}'"
                )
                # The operator can type the abort key instead of context.
                if interjection.check_and_consume()[0] == InterjectionType.ABORT:
                    abort("at the replan prompt")
                    user_aborted = True
                    break
                if user_context:
                    logger.info(f"  Operator context: {user_context}")
                new_queue = do_replan(sub_task, "The user requested a replan.",
                                      user_context, "user_replan")
                if new_queue is not None:
                    pending = new_queue
                user_replanned = True
                break

            # --- No-movement detection ---
            no_movement = (ep_result.converged
                           and ep_result.max_displacement < cfg.no_movement_threshold)

            if no_movement:
                reason = (
                    f"The robot did not move (max_displacement="
                    f"{ep_result.max_displacement:.4f}). It likely did not "
                    f"understand the object name in the instruction."
                )
                logger.warning(reason)
                break  # skip remaining retries, fall through to replan

            if not cfg.evaluate_subtasks:
                succeeded = True
                break

            # 3. Judge success — from a video clip of the attempt if available,
            #    otherwise a fresh still frame.
            clip_fps = None
            if cfg.vlm_eval_use_video:
                observation, clip_span = frames.capture_clip(
                    tag=f"eval_task{task_num}",
                    num_frames=cfg.vlm_eval_num_frames,
                )
                if not observation:
                    # Empty buffer (e.g. very short episode) — fall back to a still.
                    observation = frames.capture(tag=f"eval_task{task_num}")
                    logger.info("Evaluation observation: static image "
                                "(video requested but clip buffer was empty).")
                else:
                    if clip_span > 0 and len(observation) > 1:
                        # The clip's true rate is measured over the span those frames
                        # actually cover. The buffer decimates rather than dropping
                        # its head, so the span now tracks the whole episode; deriving
                        # the rate from the measured span (rather than assuming the
                        # episode duration) keeps the model's frame timestamps honest
                        # even if the buffer started or ended slightly inside the run.
                        # n frames span n-1 intervals, which is what a rate divides:
                        # 8 frames over 14s is 0.5 fps, not 0.571.
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
            try:
                eval_result = planner.evaluate(sub_task, observation, fps=clip_fps)
            except ValueError as e:
                logger.warning(f"Could not parse VLM evaluation ({e}); "
                               "assuming success and moving on.")
                succeeded = True
                break
            except Exception as e:
                # An unexpected evaluator failure (CUDA OOM, an unsupported
                # processor kwarg) is typically systematic, so continuing would
                # march through the queue with no working success signal. Abandon
                # the high-level task rather than report unverified sub-tasks as
                # done — a run that silently claims success is worse than one that
                # stops. `completed` deliberately does not gain this sub-task.
                logger.exception(
                    f"VLM evaluation errored ({e}); abandoning high-level task "
                    f"'{prompt}' — sub-task success can no longer be verified. "
                    f"Completed and verified before the failure: {completed}"
                )
                return

            # Skipping/replanning during evaluation is not an option: the
            # judgement has already been produced, so acting on the request would
            # throw it away. Consume and discard anything that arrived while the
            # VLM was running, so it cannot leak into the next episode as a stale
            # skip the operator no longer intends. An abort is honoured, though —
            # it is about the whole task, not this one judgement.
            itype, _ = interjection.check_and_consume()
            if itype == InterjectionType.ABORT:
                abort("during evaluation")
                user_aborted = True
                break
            elif itype != InterjectionType.NONE:
                logger.info(f"Ignoring {itype.value} requested during evaluation "
                            f"— '{sub_task}' was already judged.")

            success = bool(eval_result.get("success"))
            reason = eval_result.get("reason", "")
            logger.info(f"VLM judgement: success={success} | {reason}")

            if success:
                succeeded = True
                break

            if attempts_left > 0:
                logger.info(f"Retrying sub-task ({attempts_left} attempt(s) left)...")

        if user_aborted:
            logger.info(f"  Completed before the abort: {completed}")
            return

        if user_replanned:
            continue

        if succeeded:
            completed.append(sub_task)
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

        # Give the operator a chance to tell the planner what actually went
        # wrong. This BLOCKS until Enter — an unattended run will sit here
        # until someone responds. Bare Enter replans with no extra context.
        user_context = interjection.prompt_for_context(
            f"[REPLAN] Sub-task '{sub_task}' failed after all retries: {reason}"
        )
        # The operator can type the abort key instead of context.
        if interjection.check_and_consume()[0] == InterjectionType.ABORT:
            abort("at the replan prompt")
            logger.info(f"  Completed before the abort: {completed}")
            return
        if user_context:
            logger.info(f"  Operator context: {user_context}")

        new_subtasks = do_replan(sub_task, reason, user_context, "replan")
        if new_subtasks is None:
            continue

        # Replace the pending queue with the replanned tasks
        pending = new_subtasks

    logger.info(f"High-level task complete: '{prompt}'")
    logger.info(f"  Completed {len(completed)}/{len(subtasks)} original sub-tasks: {completed}")


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
    client = LoopRobotClient(cfg)
    if not client.connect_server():
        client.logger.error("Could not connect to policy server. Exiting.")
        client.stop()
        return
    # 2. VLM planner (reasoning layer)
    planner = VLMPlanner(cfg.vlm_model, temperature=cfg.vlm_temperature,
                         eval_fps=cfg.vlm_eval_fps,
                         two_pass=cfg.vlm_two_pass_decompose)
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