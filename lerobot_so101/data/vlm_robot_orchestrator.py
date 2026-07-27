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
from vlm import (
    DECOMPOSITION_SYSTEM_PROMPT,
    EVALUATION_SYSTEM_PROMPT,
    generate,
    load_model,
)


# ---------------------------------------------------------------------------
# Config: extends the loop client config with VLM parameters
# ---------------------------------------------------------------------------
@dataclass
class OrchestratorConfig(LoopClientConfig):
    # HuggingFace model name or local path for the VLM planner
    vlm_model: str = "Qwen/Qwen3-VL-4B-Instruct"
    # Sampling temperature for VLM generation
    vlm_temperature: float = 0.7
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
    # Number of frames sampled from the episode buffer for the eval clip.
    vlm_eval_num_frames: int = 8
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


class InterjectionManager:
    """Background stdin listener that lets the user skip or replan mid-execution."""

    def __init__(self, client: "LoopRobotClient"):
        self.client = client
        self._type = InterjectionType.NONE
        self._replan_context = ""
        self._lock = threading.Lock()
        self._active = threading.Event()
        # Set while the main thread is blocking on stdin itself; the listener
        # must not consume input during that window.
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        self._clear()
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

    def prompt_for_context(self, header: str) -> str:
        """Block on stdin for one line of operator guidance and return it.

        Pauses the background listener first so it does not swallow the typed
        line. Safe to call when no listener is running (--enable_interjection
        =false): it falls back to a plain input().
        """
        listener_running = self._thread is not None and self._active.is_set()
        if not listener_running:
            print(f"\n{header}")
            try:
                return input("[REPLAN] Additional context for the planner "
                             "(Enter for none): ").strip()
            except EOFError:
                return ""

        self._paused.set()
        # Longer than the listener's select() timeout, so any in-flight poll
        # returns and observes _paused before touching stdin.
        time.sleep(0.25)
        try:
            print(f"\n{header}")
            print("[REPLAN] Additional context for the planner (Enter for none): ",
                  end="", flush=True)
            line = sys.stdin.readline()
            return line.strip() if line else ""
        finally:
            self._paused.clear()

    def _listener(self):
        print("\n[INTERJECT] Press 's'+Enter to SKIP subtask | 'r'+Enter to REPLAN | Enter to skip")
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
                if line == "r":
                    print("[INTERJECT] Why is the plan wrong? Type context and press Enter:")
                    context = sys.stdin.readline().strip()
                    if not context:
                        context = "(no context provided)"
                    with self._lock:
                        self._type = InterjectionType.REPLAN
                        self._replan_context = context
                    print(f"[INTERJECT] REPLAN requested: {context}")
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
                 eval_fps: float = 2.0):
        self.model, self.processor = load_model(model_name)
        self.temperature = temperature
        self.eval_fps = eval_fps

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
        """High-level prompt + observation -> ordered list of sub-tasks."""
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

    def __init__(self, client: LoopRobotClient, camera_key: str, save_dir: Path | None):
        self.client = client
        self.camera_key = camera_key
        self.save_dir = save_dir

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

        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = self.save_dir / f"{tag}_{ts}.png"
            frame.save(path)
            logging.info(f"Saved VLM frame: {path}")

        return frame

    def capture_clip(self, tag: str = "clip", num_frames: int = 8) -> list[Image.Image]:
        """Return the buffered episode clip as a list of PIL frames (oldest-first).

        Returns [] if the client buffered nothing (e.g. a very short episode), in
        which case the caller should fall back to a single still.
        """
        arrays = self.client.get_episode_clip(num_frames)
        clip: list[Image.Image] = []
        for arr in arrays:
            if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] == 3:
                clip.append(Image.fromarray(arr.astype(np.uint8)))

        if not clip:
            logging.warning("Episode clip buffer empty — no video for evaluation.")
            return []

        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            for i, img in enumerate(clip):
                path = self.save_dir / f"{tag}_{ts}_{i:02d}.png"
                img.save(path)
            logging.info(f"Saved VLM eval clip ({len(clip)} frames) to {self.save_dir}")

        return clip

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
):
    logger = client.logger

    # 1. Decompose with a fresh observation
    logger.info(f"Decomposing high-level prompt: '{prompt}'")
    frame = frames.capture(tag="decompose")
    try:
        subtasks = planner.decompose(prompt, frame)
    except ValueError as e:
        logger.error(str(e))
        logger.error("Decomposition failed — skipping this prompt.")
        return

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
            itype, ctx = interjection.check_and_consume()
            if itype == InterjectionType.SKIP:
                user_skipped = True
                reason = "The user skipped this episode."
                logger.info(f"User skipped sub-task: '{sub_task}' "
                            f"({attempts_left} attempt(s) left)")
                client.go_home()
                continue
            elif itype == InterjectionType.REPLAN:
                logger.info(f"User requested replan: {ctx}")
                client.go_home()
                new_queue = do_replan(sub_task, "The user requested a replan.",
                                      ctx, "user_replan")
                if new_queue is not None:
                    pending = new_queue
                user_replanned = True
                break

            # Return home so the arm is out of frame and the next episode
            # starts from a consistent pose
            client.go_home()

            # --- Interjection check: after go_home ---
            itype, ctx = interjection.check_and_consume()
            if itype == InterjectionType.SKIP:
                user_skipped = True
                reason = "The user skipped this episode."
                logger.info(f"User skipped sub-task: '{sub_task}' "
                            f"({attempts_left} attempt(s) left)")
                continue
            elif itype == InterjectionType.REPLAN:
                logger.info(f"User requested replan: {ctx}")
                new_queue = do_replan(sub_task, "The user requested a replan.",
                                      ctx, "user_replan")
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
                observation = frames.capture_clip(
                    tag=f"eval_task{task_num}",
                    num_frames=cfg.vlm_eval_num_frames,
                )
                if not observation:
                    # Empty buffer (e.g. very short episode) — fall back to a still.
                    observation = frames.capture(tag=f"eval_task{task_num}")
                    logger.info("Evaluation observation: static image "
                                "(video requested but clip buffer was empty).")
                else:
                    if ep_result.duration > 0:
                        # The buffer is sampled across the whole episode, so the
                        # clip's true rate is frames / episode duration — not the
                        # nominal vlm_eval_fps. Feeding the real rate makes the
                        # model's frame timestamps span the actual attempt.
                        clip_fps = len(observation) / ep_result.duration
                    logger.info(
                        f"Evaluation observation: video clip "
                        f"({len(observation)} frames over "
                        f"{ep_result.duration:.1f}s, "
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
                # A broken evaluator must not tear down the robot session, nor
                # burn retries/replans on every sub-task. Log the full trace so
                # a systematic failure is still obvious.
                logger.exception(f"VLM evaluation errored ({e}); "
                                 "assuming success and moving on.")
                succeeded = True
                break

            # Interjecting during evaluation is not an option: the judgement
            # has already been produced, so acting on the request would throw
            # it away. Consume and discard anything that arrived while the VLM
            # was running, so it cannot leak into the next episode as a stale
            # skip the operator no longer intends.
            itype, _ = interjection.check_and_consume()
            if itype != InterjectionType.NONE:
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
# Main entry point
# ---------------------------------------------------------------------------
@draccus.wrap()
def main(cfg: OrchestratorConfig):
    logging.basicConfig(level=logging.INFO)
    logging.info(pformat(asdict(cfg)))

    # 1. Robot + policy server (VLA layer)
    client = LoopRobotClient(cfg)
    if not client.connect_server():
        client.logger.error("Could not connect to policy server. Exiting.")
        client.stop()
        return

    # 2. VLM planner (reasoning layer)
    planner = VLMPlanner(cfg.vlm_model, temperature=cfg.vlm_temperature,
                         eval_fps=cfg.vlm_eval_fps)
    # 3. Frame source for the VLM
    save_dir = Path("./frames") if cfg.save_frames else None
    frames = VLMFrameSource(client, camera_key=cfg.vlm_camera_key, save_dir=save_dir)
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

            if cfg.enable_interjection:
                interjection.start()
            run_high_level_task(client, planner, frames, cfg, prompt, interjection)
            if cfg.enable_interjection:
                interjection.stop()

    except KeyboardInterrupt:
        client.logger.info("\nInterrupted by user.")

    finally:
        client.go_home()
        frames.stop()
        client.stop()


if __name__ == "__main__":
    register_third_party_plugins()
    main()