"""
Evaluate Qwen3-VL-8B-Instruct as the VLM reasoning layer using vLLM.

The 8B model does not support HuggingFace transformers generate() —
this script uses vLLM for inference instead.

Tests the same capabilities as eval_qwen3vl.py:
1. Task decomposition: natural language prompt + images -> sub-task queue
2. Preference sensitivity: different prompts -> different decompositions
3. Success evaluation: image + sub-task -> success/failure judgement

Requirements:
    pip install vllm qwen-vl-utils pyrealsense2 pillow numpy

Usage:
    # Live capture from RealSense D435 (default):
    python eval_qwen3vl_8b.py

    # Static image:
    python eval_qwen3vl_8b.py --images top.png

    # Custom model path or quantisation:
    python eval_qwen3vl_8b.py --model Qwen/Qwen3-VL-8B-Instruct-FP8

    # Adjust GPU memory usage (default 0.70):
    python eval_qwen3vl_8b.py --gpu-mem 0.80

    # Text-only (no camera):
    python eval_qwen3vl_8b.py --no-camera
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from pathlib import Path
from PIL import Image

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# RealSense D435 capture (identical to vlm.py)
# ---------------------------------------------------------------------------

class RealSenseCamera:
    """Wrapper around the Intel RealSense D435 for RGB frame capture."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(
            rs.stream.color, width, height, rs.format.rgb8, fps
        )
        self.width = width
        self.height = height
        self.started = False

    def start(self):
        if not self.started:
            self.pipeline.start(self.config)
            for _ in range(30):
                self.pipeline.wait_for_frames()
            self.started = True
            print(f"RealSense D435 started ({self.width}x{self.height})")

    def capture(self) -> Image.Image:
        if not self.started:
            self.start()
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Failed to capture colour frame from RealSense")
        return Image.fromarray(np.asanyarray(color_frame.get_data()))

    def stop(self):
        if self.started:
            self.pipeline.stop()
            self.started = False
            print("RealSense D435 stopped")

    def __del__(self):
        self.stop()


def capture_scene(camera: RealSenseCamera, save_dir: Path = None) -> Image.Image:
    frame = camera.capture()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = save_dir / f"frame_top_{ts}.png"
        frame.save(path)
        print(f"  Saved frame: {path}")
    return frame


# ---------------------------------------------------------------------------
# System prompts (shared with vlm.py / orchestrator)
# ---------------------------------------------------------------------------

DECOMPOSITION_SYSTEM_PROMPT = """You are a robot task planner. You receive a user's natural language instruction and a camera observation from an orange tabletop robot arm (SO-101, 6-DOF).

Your job is to decompose the instruction into a numbered sequence of simple manipulation sub-tasks that the robot can execute one at a time.

You are controlling the orange robot arm in the frame. The robot arm itself is not an object in the scene — exclude it when identifying objects to manipulate.

Rules:
- Before generating sub-tasks, first identify any preferences or constraints in the instruction (e.g. objects to exclude, ordering requirements). Then identify which visible objects match those constraints. Only after this reasoning should you produce the sub-task list.
- Each sub-task must be a single, atomic manipulation action (e.g. "grab the orange and put it on the tray").
- Use simple, concrete language. Avoid abstract or vague instructions.
- Pick-and-place is a single atomic action. Do not separate picking up and placing into two sub-tasks. Use a single instruction like "put the X on the Y" rather than "grab the X" followed by "place the X on the Y".
- Ground sub-tasks in what you observe in the image. Only reference objects you can see.
- If the instruction specifies a destination (e.g. "on the tray", "in the bowl"), that destination is not an object to be moved — do not generate a sub-task to move it.
- If the user's instruction contains preferences (e.g. "leave the cups", "no sugar", "put the red one first"), reflect those preferences in which sub-tasks you include, omit, or reorder.
- Do NOT include sub-tasks that violate stated preferences.
- If the instruction refers to a group of objects using words like "everything", "all", "the rest", or similar, visually identify each individual object in the scene and generate one sub-task per object. Do not output an empty list — if objects are visible, there is work to do.
- If the instruction specifies to leave something or ignore something, do not output any subtasks that involve the specified object.

Output format:
Return ONLY a JSON array of sub-task strings. Example:
["pick up the orange", "place the orange in the bowl"]
"""

EVALUATION_SYSTEM_PROMPT = """You are a robot task evaluator. You receive a camera observation and a sub-task that was just attempted by a robot arm.

Assess whether the sub-task was completed successfully based on the visual evidence.

Output format:
Return ONLY a JSON object with two fields:
- "success": true or false
- "reason": a brief explanation of your judgement

Example:
{"success": true, "reason": "The orange is now inside the bowl as instructed."}
"""


# ---------------------------------------------------------------------------
# vLLM model wrapper
# ---------------------------------------------------------------------------

class VLLMModel:
    """Wraps vLLM LLM + processor for Qwen3-VL inference."""

    def __init__(self, model_name: str, gpu_memory_utilization: float = 0.70):
        print(f"Loading {model_name} with vLLM...")
        t0 = time.time()

        self.processor = AutoProcessor.from_pretrained(model_name)

        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=False,
            tensor_parallel_size=torch.cuda.device_count(),
            seed=0,
        )

        print(f"Model loaded in {time.time() - t0:.1f}s")
        print(f"GPUs: {torch.cuda.device_count()}")

    def generate(self, messages: list, temperature: float = 0.7,
                 max_new_tokens: int = 1024) -> str:
        """Run inference on a single message set and return generated text."""

        # Build the prompt string via the processor's chat template
        prompt_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Extract image/video inputs using qwen_vl_utils
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        vllm_input = {
            "prompt": prompt_text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_k=20,
            top_p=0.8,
        )

        t0 = time.time()
        outputs = self.llm.generate([vllm_input], sampling_params=sampling_params)
        elapsed = time.time() - t0

        generated_text = outputs[0].outputs[0].text
        n_tokens = len(outputs[0].outputs[0].token_ids)
        print(f"  Generated {n_tokens} tokens in {elapsed:.1f}s "
              f"({n_tokens / elapsed:.1f} tok/s)")

        return generated_text


# ---------------------------------------------------------------------------
# Parsing helpers (shared logic with orchestrator)
# ---------------------------------------------------------------------------

import re


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences the model sometimes wraps JSON in."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_subtask_list(output: str) -> list[str]:
    """Extract a JSON array of sub-task strings from model output."""
    text = _strip_think_tags(output)
    text = _strip_code_fences(text)

    candidates = [text]
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list) and all(isinstance(t, str) for t in parsed):
                if not parsed:
                    raise ValueError(
                        f"VLM returned an empty sub-task list:\n{output}"
                    )
                return parsed
        except json.JSONDecodeError:
            continue

    raise ValueError(f"Could not parse sub-task list from VLM output:\n{output}")


def parse_evaluation(output: str) -> dict:
    """Extract a {'success': bool, 'reason': str} object from model output."""
    text = _strip_think_tags(output)
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
# Image content helpers
# ---------------------------------------------------------------------------

def get_image_content(camera=None, image_paths=None, save_dir=None):
    """Get image content from either live camera or file paths.

    Returns (content_list, description_string) or (None, None).
    """
    if camera is not None:
        frame = capture_scene(camera, save_dir=save_dir)
        return [
            {"type": "text", "text": "[top-down camera]"},
            {"type": "image", "image": frame},
        ], "live RealSense capture"

    elif image_paths:
        content = []
        labels = ["top-down camera", "side camera"]
        for i, path in enumerate(image_paths):
            label = labels[i] if i < len(labels) else f"camera {i + 1}"
            content.append({"type": "text", "text": f"[{label}]"})
            content.append({"type": "image", "image": str(path)})
        return content, str([str(p) for p in image_paths])

    return None, None


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

INTERACTIVE_HELP = """
Commands:
  <any text>          Decompose a prompt (e.g. "clean up the table")
  /eval <sub-task>    Evaluate a specific sub-task against current camera frame
  /temp <value>       Set temperature (current: {temp})
  /save               Toggle saving frames to disk (current: {save})
  /help               Show this help
  /quit               Exit
""".strip()


def interactive_loop(vllm_model: VLLMModel, camera, image_paths, save_dir):
    """Interactive REPL — model stays loaded, type prompts freely."""

    temp = 0.7
    saving = save_dir is not None

    print(f"\n{'=' * 60}")
    print("INTERACTIVE MODE (vLLM / 8B) — type prompts to decompose.")
    print("Type /help for commands, /quit to exit.")
    print(f"{'=' * 60}")

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        # --- Commands ---
        if user_input in ("/quit", "/exit"):
            print("Exiting.")
            break

        elif user_input == "/help":
            print(INTERACTIVE_HELP.format(
                temp=temp,
                save="ON" if saving else "OFF",
            ))

        elif user_input.startswith("/eval "):
            sub_task = user_input[6:].strip()
            if not sub_task:
                print("Usage: /eval <sub-task description>")
                continue

            image_content, img_desc = get_image_content(
                camera, image_paths, save_dir if saving else None
            )
            user_content = []
            if image_content:
                user_content.extend(image_content)
            else:
                user_content.append({
                    "type": "text",
                    "text": "The robot arm is at a tabletop.",
                })
            user_content.append({
                "type": "text",
                "text": (
                    f'\nThe robot just attempted this sub-task: "{sub_task}"\n'
                    "Did it succeed?"
                ),
            })

            messages = [
                {"role": "system", "content": [{"type": "text", "text": EVALUATION_SYSTEM_PROMPT}]},
                {"role": "user", "content": user_content},
            ]

            output = vllm_model.generate(messages, temperature=temp)
            print(f"\nSub-task: \"{sub_task}\"")
            print(f"Raw output:\n{output}")

            try:
                result = parse_evaluation(output)
                print(f"Parsed: success={result.get('success')}, "
                      f"reason={result.get('reason')}")
            except ValueError as e:
                print(f"[WARNING] {e}")

        elif user_input.startswith("/temp "):
            try:
                temp = float(user_input[6:].strip())
                print(f"Temperature set to {temp}")
            except ValueError:
                print("Usage: /temp <float>  (e.g. /temp 0.3)")

        elif user_input == "/save":
            saving = not saving
            print(f"Frame saving: {'ON — saving to ./frames/' if saving else 'OFF'}")

        elif user_input.startswith("/"):
            print(f"Unknown command: {user_input}. Type /help for options.")

        # --- Decomposition prompt ---
        else:
            image_content, img_desc = get_image_content(
                camera, image_paths, save_dir if saving else None
            )

            print(f"\n{'=' * 60}")
            print("TASK DECOMPOSITION TEST")
            print(f"Prompt: \"{user_input}\"")
            print(f"Image source: {img_desc or 'text-only'}")
            print(f"{'=' * 60}")

            user_content = []
            if image_content:
                user_content.extend(image_content)
                user_content.append({
                    "type": "text",
                    "text": f"\nInstruction: {user_input}",
                })
            else:
                user_content.append({
                    "type": "text",
                    "text": (
                        "No camera observation is available. Decompose the "
                        "instruction based on the text alone.\n\n"
                        f"Instruction: {user_input}"
                    ),
                })

            messages = [
                {"role": "system", "content": [{"type": "text", "text": DECOMPOSITION_SYSTEM_PROMPT}]},
                {"role": "user", "content": user_content},
            ]

            output = vllm_model.generate(messages, temperature=temp)
            print(f"\nRaw output:\n{output}")

            try:
                tasks = parse_subtask_list(output)
                print(f"\nParsed {len(tasks)} sub-tasks:")
                for i, task in enumerate(tasks, 1):
                    print(f"  {i}. {task}")
            except ValueError as e:
                print(f"\n[WARNING] {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL-8B for robot task reasoning (vLLM)"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model name or path (default: Qwen/Qwen3-VL-8B-Instruct)",
    )
    parser.add_argument(
        "--images", nargs="*", default=None,
        help="Paths to camera image files (overrides live camera)",
    )
    parser.add_argument(
        "--no-camera", action="store_true",
        help="Disable RealSense camera, use text-only fallback",
    )
    parser.add_argument(
        "--save-frames", action="store_true",
        help="Save captured camera frames to ./frames/",
    )
    parser.add_argument(
        "--gpu-mem", type=float, default=0.70,
        help="vLLM gpu_memory_utilization (default: 0.70)",
    )
    parser.add_argument(
        "--resolution", nargs=2, type=int, default=[640, 480],
        metavar=("W", "H"),
        help="RealSense capture resolution (default: 640 480)",
    )
    args = parser.parse_args()

    # --- Resolve image source ---
    camera = None
    image_paths = None
    save_dir = Path("./frames") if args.save_frames else None

    if args.images:
        image_paths = [Path(p) for p in args.images]
        for p in image_paths:
            if not p.exists():
                print(f"[ERROR] Image not found: {p}")
                return
        print(f"Using static images: {[str(p) for p in image_paths]}")

    elif not args.no_camera:
        try:
            camera = RealSenseCamera(
                width=args.resolution[0],
                height=args.resolution[1],
            )
            camera.start()
        except Exception as e:
            print(f"[WARNING] Could not start RealSense camera: {e}")
            print("Falling back to text-only mode.")
            camera = None

    if camera is None and image_paths is None and not args.no_camera:
        print("[INFO] No image source available — running in text-only mode")

    # --- Load model ---
    vllm_model = VLLMModel(args.model, gpu_memory_utilization=args.gpu_mem)

    # --- Run ---
    interactive_loop(vllm_model, camera, image_paths, save_dir)

    if camera is not None:
        camera.stop()


if __name__ == "__main__":
    main()