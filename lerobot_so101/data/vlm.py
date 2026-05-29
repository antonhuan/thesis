"""
Evaluate Qwen3-VL-2B-Instruct as the VLM reasoning layer.

Tests three capabilities needed for the dual-system architecture:
1. Task decomposition: natural language prompt + images -> sub-task queue
2. Preference sensitivity: different prompts -> different decompositions
3. Success evaluation: image + sub-task -> success/failure judgement

Captures frames directly from an Intel RealSense D435 camera (top-down view).

Requirements:
    pip install torch transformers accelerate pillow pyrealsense2 numpy

    # Qwen3-VL requires latest transformers (built from source or >= 4.57.0)
    pip install git+https://github.com/huggingface/transformers

Usage:
    # Live capture from RealSense D435 (default):
    python eval_qwen3vl.py

    # Fall back to static image files if no camera:
    python eval_qwen3vl.py --images top.png

    # Custom prompt:
    python eval_qwen3vl.py --prompt "pick up the red cup and place it on the left side"

    # Save captured frames to disk:
    python eval_qwen3vl.py --save-frames

    # Skip camera, text-only:
    python eval_qwen3vl.py --no-camera
"""

import argparse
import time
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------------------------
# RealSense D435 capture
# ---------------------------------------------------------------------------

class RealSenseCamera:
    """Wrapper around the Intel RealSense D435 for RGB frame capture."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        import pyrealsense2 as rs

        self.rs = rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Only enable colour stream — we don't need depth for VLM input
        self.config.enable_stream(
            rs.stream.color, width, height, rs.format.rgb8, fps
        )

        self.width = width
        self.height = height
        self.started = False

    def start(self):
        """Start the camera pipeline."""
        if not self.started:
            profile = self.pipeline.start(self.config)
            # Let auto-exposure settle
            for _ in range(30):
                self.pipeline.wait_for_frames()
            self.started = True
            print(f"RealSense D435 started ({self.width}x{self.height})")

    def capture(self) -> Image.Image:
        """Capture a single RGB frame and return as PIL Image."""
        if not self.started:
            self.start()

        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            raise RuntimeError("Failed to capture colour frame from RealSense")

        color_array = np.asanyarray(color_frame.get_data())  # H x W x 3 (RGB)
        return Image.fromarray(color_array)

    def stop(self):
        """Stop the camera pipeline."""
        if self.started:
            self.pipeline.stop()
            self.started = False
            print("RealSense D435 stopped")

    def __del__(self):
        self.stop()


def capture_scene(camera: RealSenseCamera, save_dir: Path = None) -> Image.Image:
    """Capture a frame from the top-down RealSense camera.
    
    Optionally saves the frame to disk with a timestamp.
    """
    frame = camera.capture()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = save_dir / f"frame_top_{ts}.png"
        frame.save(path)
        print(f"  Saved frame: {path}")

    return frame


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------

def load_model(model_name: str = "Qwen/Qwen3-VL-4B-Instruct"):
    """Load model and processor."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    print(f"Loading {model_name}...")
    t0 = time.time()

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"Device: {model.device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    return model, processor


def generate(model, processor, messages: list, max_new_tokens: int = 1024,
             temperature: float = 0.7) -> str:
    """Run inference and return generated text."""
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    t0 = time.time()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.8,
            top_k=20,
            temperature=temperature,
        )

    # Trim input tokens from output
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    elapsed = time.time() - t0
    n_tokens = len(generated_ids_trimmed[0])
    print(f"  Generated {n_tokens} tokens in {elapsed:.1f}s ({n_tokens/elapsed:.1f} tok/s)")

    return output_text


# ---------------------------------------------------------------------------
# System prompts matching the dual-system architecture
# ---------------------------------------------------------------------------

DECOMPOSITION_SYSTEM_PROMPT = """You are a robot task planner. You receive a user's natural language instruction and a camera observation from a tabletop robot arm (SO-101, 6-DOF).

Your job is to decompose the instruction into a numbered sequence of simple manipulation sub-tasks that the robot can execute one at a time.

Rules:
- Each sub-task must be a single, atomic manipulation action (e.g. "pick up the orange", "place the cup on the left side of the table").
- Use simple, concrete language. Avoid abstract or vague instructions.
- Ground sub-tasks in what you observe in the image. Only reference objects you can see.
- If the user's instruction contains preferences (e.g. "leave the cups", "no sugar", "put the red one first"), reflect those preferences in which sub-tasks you include, omit, or reorder.
- Do NOT include sub-tasks that violate stated preferences.

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
# Image content builders
# ---------------------------------------------------------------------------

def build_image_content_pil(image: Image.Image) -> list[dict]:
    """Build content list from a PIL Image (live camera capture)."""
    return [
        {"type": "text", "text": "[top-down camera]"},
        {"type": "image", "image": image},
    ]


def build_image_content_paths(image_paths: list[Path]) -> list[dict]:
    """Build content list from file paths."""
    content = []
    camera_labels = ["top-down camera", "side camera"]

    for i, path in enumerate(image_paths):
        label = camera_labels[i] if i < len(camera_labels) else f"camera {i+1}"
        content.append({"type": "text", "text": f"[{label}]"})
        content.append({"type": "image", "image": str(path)})

    return content


def get_image_content(camera=None, image_paths=None, save_dir=None):
    """Get image content from either live camera or file paths.
    
    Returns (content_list, description_string) or (None, None) if no source.
    """
    if camera is not None:
        frame = capture_scene(camera, save_dir=save_dir)
        return build_image_content_pil(frame), "live RealSense capture"
    elif image_paths:
        return build_image_content_paths(image_paths), str([str(p) for p in image_paths])
    else:
        return None, None


# ---------------------------------------------------------------------------
# Test routines
# ---------------------------------------------------------------------------

def test_decomposition(model, processor, prompt: str,
                       camera=None, image_paths=None, save_dir=None):
    """Test task decomposition with camera or image files."""
    image_content, img_desc = get_image_content(camera, image_paths, save_dir)

    print(f"\n{'='*60}")
    print(f"TASK DECOMPOSITION TEST")
    print(f"Prompt: \"{prompt}\"")
    print(f"Image source: {img_desc or 'text-only'}")
    print(f"{'='*60}")

    user_content = []

    if image_content:
        user_content.extend(image_content)
        user_content.append({"type": "text", "text": f"\nInstruction: {prompt}"})
    else:
        user_content.append({
            "type": "text",
            "text": (
                "The robot is at a tabletop with an orange, a blue cup, "
                "a red cup, and a plate.\n\n"
                f"Instruction: {prompt}"
            ),
        })

    messages = [
        {"role": "system", "content": [{"type": "text", "text": DECOMPOSITION_SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]

    output = generate(model, processor, messages)
    print(f"\nModel output:\n{output}")

    # Try to parse as JSON
    try:
        tasks = json.loads(output.strip())
        print(f"\nParsed {len(tasks)} sub-tasks:")
        for i, task in enumerate(tasks, 1):
            print(f"  {i}. {task}")
    except json.JSONDecodeError:
        print("\n[WARNING] Output is not valid JSON — may need prompt tuning")

    return output


def test_preference_sensitivity(model, processor,
                                camera=None, image_paths=None, save_dir=None):
    """Test whether different preferences produce different decompositions."""
    print(f"\n{'='*60}")
    print(f"PREFERENCE SENSITIVITY TEST")
    print(f"{'='*60}")

    prompt_pairs = [
        (
            "clean up the table",
            "clean up the table but leave the cups",
        ),
        (
            "pick up the orange and place it in the cup",
            "pick up the orange and place it in the red cup",
        ),
        (
            "move everything to the left side",
            "move everything to the left side, starting with the plate",
        ),
    ]

    for base_prompt, pref_prompt in prompt_pairs:
        print(f"\n--- Pair ---")
        print(f"  Baseline:   \"{base_prompt}\"")
        out_base = test_decomposition(
            model, processor, base_prompt, camera, image_paths, save_dir
        )

        print(f"  Preference: \"{pref_prompt}\"")
        out_pref = test_decomposition(
            model, processor, pref_prompt, camera, image_paths, save_dir
        )

        different = out_base.strip() != out_pref.strip()
        print(f"\n  Outputs differ: {different}")


def test_success_evaluation(model, processor,
                            camera=None, image_paths=None, save_dir=None):
    """Test success/failure judgement capability."""
    image_content, img_desc = get_image_content(camera, image_paths, save_dir)

    print(f"\n{'='*60}")
    print(f"SUCCESS EVALUATION TEST")
    print(f"Image source: {img_desc or 'text-only'}")
    print(f"{'='*60}")

    sub_task = "pick up the orange"

    user_content = []
    if image_content:
        user_content.extend(image_content)
    else:
        user_content.append({
            "type": "text",
            "text": "The robot arm is at its home position. An orange is sitting on the table.",
        })

    user_content.append({
        "type": "text",
        "text": f"\nThe robot just attempted this sub-task: \"{sub_task}\"\nDid it succeed?",
    })

    messages = [
        {"role": "system", "content": [{"type": "text", "text": EVALUATION_SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]

    output = generate(model, processor, messages)
    print(f"\nSub-task: \"{sub_task}\"")
    print(f"Model output:\n{output}")

    try:
        result = json.loads(output.strip())
        print(f"\nParsed: success={result.get('success')}, reason={result.get('reason')}")
    except json.JSONDecodeError:
        print("\n[WARNING] Output is not valid JSON")

    return output


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

INTERACTIVE_HELP = """
Commands:
  <any text>          Decompose a prompt (e.g. "clean up the table")
  /preference         Run the preference sensitivity test suite
  /evaluate           Run the success evaluation test
  /eval <sub-task>    Evaluate a specific sub-task against current camera frame
  /temp <value>       Set temperature (current: {temp})
  /save               Toggle saving frames to disk (current: {save})
  /help               Show this help
  /quit               Exit
""".strip()


def interactive_loop(model, processor, camera, image_paths, save_dir):
    """Interactive REPL — model stays loaded, type prompts freely."""

    temp = 0.7
    saving = save_dir is not None

    print(f"\n{'='*60}")
    print("INTERACTIVE MODE — model loaded, type prompts to decompose.")
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

        # --- Commands ---
        if user_input == "/quit" or user_input == "/exit":
            print("Exiting.")
            break

        elif user_input == "/help":
            print(INTERACTIVE_HELP.format(
                temp=temp,
                save="ON" if saving else "OFF",
            ))

        elif user_input == "/preference":
            test_preference_sensitivity(
                model, processor, camera, image_paths,
                save_dir if saving else None,
            )

        elif user_input == "/evaluate":
            test_success_evaluation(
                model, processor, camera, image_paths,
                save_dir if saving else None,
            )

        elif user_input.startswith("/eval "):
            sub_task = user_input[6:].strip()
            if not sub_task:
                print("Usage: /eval <sub-task description>")
                continue
            # Run a custom success evaluation
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
                "text": f'\nThe robot just attempted this sub-task: "{sub_task}"\nDid it succeed?',
            })
            messages = [
                {"role": "system", "content": [{"type": "text", "text": EVALUATION_SYSTEM_PROMPT}]},
                {"role": "user", "content": user_content},
            ]
            output = generate(model, processor, messages)
            print(f"\nSub-task: \"{sub_task}\"")
            print(f"Model output:\n{output}")
            try:
                result = json.loads(output.strip())
                print(f"Parsed: success={result.get('success')}, reason={result.get('reason')}")
            except json.JSONDecodeError:
                print("[WARNING] Output is not valid JSON")

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
            test_decomposition(
                model, processor, user_input, camera, image_paths,
                save_dir if saving else None,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-VL-2B for robot task reasoning"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-VL-4B-Instruct",
        help="HuggingFace model name or local path",
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
        "--prompt", default=None,
        help="Custom prompt for single decomposition test",
    )
    parser.add_argument(
        "--test", choices=["decompose", "preference", "evaluate", "all"],
        default="all",
        help="Which test to run",
    )
    parser.add_argument(
        "--resolution", nargs=2, type=int, default=[640, 480],
        metavar=("W", "H"),
        help="RealSense capture resolution (default: 640 480)",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Enter interactive loop after loading model (keeps model in VRAM)",
    )
    args = parser.parse_args()

    # --- Resolve image source ---
    camera = None
    image_paths = None
    save_dir = Path("./frames") if args.save_frames else None

    if args.images:
        # Static image files provided
        image_paths = [Path(p) for p in args.images]
        for p in image_paths:
            if not p.exists():
                print(f"[ERROR] Image not found: {p}")
                return
        print(f"Using static images: {[str(p) for p in image_paths]}")

    elif not args.no_camera:
        # Try to initialise RealSense
        try:
            camera = RealSenseCamera(
                width=args.resolution[0],
                height=args.resolution[1],
            )
            camera.start()
        except Exception as e:
            print(f"[WARNING] Could not start RealSense camera: {e}")
            print("Falling back to text-only mode. Use --images or --no-camera.")
            camera = None

    if camera is None and image_paths is None and not args.no_camera:
        print("[INFO] No image source available — running in text-only mode")

    # --- Load model ---
    model, processor = load_model(args.model)

    # --- Run initial test if --prompt or --test provided ---
    try:
        if args.prompt:
            test_decomposition(
                model, processor, args.prompt, camera, image_paths, save_dir
            )

        elif args.test != "all" or not args.interactive:
            if args.test in ("decompose", "all"):
                test_decomposition(
                    model, processor,
                    "pick up the orange and place it in the blue cup",
                    camera, image_paths, save_dir,
                )

            if args.test in ("preference", "all"):
                test_preference_sensitivity(
                    model, processor, camera, image_paths, save_dir
                )

            if args.test in ("evaluate", "all"):
                test_success_evaluation(
                    model, processor, camera, image_paths, save_dir
                )

        # --- Interactive loop ---
        if args.interactive:
            interactive_loop(model, processor, camera, image_paths, save_dir)
        else:
            print(f"\n{'='*60}")
            print("Done. (Tip: use --interactive to keep the model loaded)")

    finally:
        if camera is not None:
            camera.stop()


if __name__ == "__main__":
    main()