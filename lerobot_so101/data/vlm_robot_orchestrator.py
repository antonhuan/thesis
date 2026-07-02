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
#   [VLM: success evaluation]  -- optional retry on failure
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
#       --max_retries=1
#
# At the prompt, type a HIGH-LEVEL instruction (e.g. "clean up the table but
# leave the cups"). The VLM decomposes it into sub-tasks, each sub-task is
# executed by the VLA policy, and (optionally) the VLM judges success from a
# fresh camera frame, retrying failed sub-tasks.
#
# Type 'quit' or 'exit' to shut down cleanly.

import json
import logging
import re
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
    # How many times to re-run a sub-task the VLM judged as failed
    max_retries: int = 1
    # Save every frame sent to the VLM under ./frames/
    save_frames: bool = False


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
    text = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL).strip()
    text = _strip_code_fences(text)

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
# VLM planner: wraps decomposition and success evaluation
# ---------------------------------------------------------------------------
class VLMPlanner:
    """Holds the loaded VLM and exposes decompose() and evaluate()."""

    def __init__(self, model_name: str, temperature: float = 0.7):
        self.model, self.processor = load_model(model_name)
        self.temperature = temperature

    def _user_content(self, frame: Image.Image | None, text: str) -> list[dict]:
        content = []
        if frame is not None:
            content.append({"type": "text", "text": "[top-down camera]"})
            content.append({"type": "image", "image": frame})
        content.append({"type": "text", "text": text})
        return content

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

    def evaluate(self, sub_task: str, frame: Image.Image | None) -> dict:
        """Observation + attempted sub-task -> {'success': bool, 'reason': str}."""
        text = (
            f'\nThe robot just attempted this sub-task: "{sub_task}"\nDid it succeed?'
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": EVALUATION_SYSTEM_PROMPT}]},
            {"role": "user", "content": self._user_content(frame, text)},
        ]
        output = generate(
            self.model, self.processor, messages, temperature=self.temperature
        )
        return parse_evaluation(output)


# ---------------------------------------------------------------------------
# Frame source for the VLM
# ---------------------------------------------------------------------------
# Replace the entire VLMFrameSource class:

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

    # 2. Execute each sub-task as a VLA episode
    for i, sub_task in enumerate(subtasks, 1):
        attempts_left = 1 + max(cfg.max_retries, 0)

        while attempts_left > 0:
            attempts_left -= 1

            logger.info(f"[{i}/{len(subtasks)}] Executing sub-task: '{sub_task}'")
            client.run_episode(sub_task)

            # Return home so the arm is out of frame and the next episode
            # starts from a consistent pose
            client.go_home()

            if not cfg.evaluate_subtasks:
                break

            # 3. Judge success from a fresh frame
            frame = frames.capture(tag=f"eval_subtask{i}")
            try:
                result = planner.evaluate(sub_task, frame)
            except ValueError as e:
                logger.warning(f"Could not parse VLM evaluation ({e}); "
                               "assuming success and moving on.")
                break

            success = bool(result.get("success"))
            reason = result.get("reason", "")
            logger.info(f"VLM judgement: success={success} | {reason}")

            if success:
                break
            if attempts_left > 0:
                logger.info(f"Retrying sub-task ({attempts_left} attempt(s) left)...")
            else:
                logger.warning(f"Sub-task '{sub_task}' failed after all attempts — "
                               "continuing with the next sub-task.")

    logger.info(f"High-level task complete: '{prompt}'")


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
    planner = VLMPlanner(cfg.vlm_model, temperature=cfg.vlm_temperature)
    # 3. Frame source for the VLM
    save_dir = Path("./frames") if cfg.save_frames else None
    frames = VLMFrameSource(client, camera_key=cfg.vlm_camera_key, save_dir=save_dir)

    client.logger.info("=" * 60)
    client.logger.info("Dual-system ready: VLM planner + VLA policy loaded.")
    client.logger.info(f"Episode duration per sub-task: {cfg.episode_duration}s")
    client.logger.info("Type a HIGH-LEVEL instruction and press Enter to execute.")
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

            run_high_level_task(client, planner, frames, cfg, prompt)

    except KeyboardInterrupt:
        client.logger.info("\nInterrupted by user.")

    finally:
        client.go_home()
        frames.stop()
        client.stop()


if __name__ == "__main__":
    register_third_party_plugins()
    main()
