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
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

    print(f"Loading {model_name}...")
    t0 = time.time()

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # model = Qwen3VLForConditionalGeneration.from_pretrained(
    #     model_name,
    #     quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    #     device_map="auto",
    # )
    processor = AutoProcessor.from_pretrained(model_name)

    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"Device: {model.device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    return model, processor


def generate(model, processor, messages: list, max_new_tokens: int = 2048,
             temperature: float = 0.7) -> str:
    """Run inference and return generated text.

    Uses qwen_vl_utils.process_vision_info so both image and video message content
    work. For image-only messages (decompose/replan) video_inputs is None, so the
    behaviour is identical to a plain image request.
    """
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        **video_kwargs,
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

# DECOMPOSITION_SYSTEM_PROMPT = """You are a task planner for an orange tabletop robot arm (SO-101, 6-DOF). You receive a natural language instruction and a camera observation of the workspace.

# Follow these steps in order:

# Step 1 — Identify objects in the scene.
# List every visible object on the table. The orange robot arm is NOT an object — exclude it. If the instruction mentions a destination (e.g. "tray", "bowl"), that is a landmark, not an object to move.

# Step 2 — Parse the instruction for preferences and constraints.
# Identify any of the following:
# - Objects to exclude or leave alone (e.g. "leave the cups", "not the banana")
# - Objects to prioritise or do first (e.g. "start with the red one", "put the apple first")
# - Ordering requirements (e.g. "before", "after", "then")
# - Conditional logic (e.g. "if there's a spoon, move it too")
# - Group references (e.g. "everything", "all", "the rest") — resolve these to the specific objects identified in Step 1
# If no preferences are found, note that and proceed with all visible objects.

# Step 3 — Build the sub-task list.
# For each object that should be acted on (i.e. not excluded by Step 2), generate one sub-task. Apply these rules:
# - Each sub-task is a single atomic pick-and-place action. Do NOT split picking and placing into separate steps. Write "put the X on the Y", not "grab X" then "place X on Y".
# - Use simple, concrete language matching what you see (e.g. "put the banana on the tray").
# - Order sub-tasks according to any sequencing preferences found in Step 2. If no order is specified, use any reasonable order.
# - Do NOT include any sub-task involving an object that the instruction says to leave, skip, or ignore.
# - If the instruction refers to "everything" or "all", generate one sub-task per visible object (minus any exclusions).

# Step 4 — Verify.
# Before outputting, check:
# - Does any sub-task involve an excluded object? If yes, remove it.
# - Does the ordering match any stated preference? If not, reorder.
# - Is every sub-task a single atomic action? If not, merge or simplify.

# Output format:
# Return ONLY a JSON array of sub-task strings. No explanation, no numbering, no markdown.
# Example: ["put the apple on the tray", "put the banana on the tray"]
# """
DECOMPOSITION_SYSTEM_PROMPT = """You are a task planner for an orange tabletop robot arm (SO-101, 6-DOF). You receive a natural language instruction and a camera observation.

You MUST respond in the following JSON format exactly:

{
  "visible_objects": ["list", "of", "objects", "you", "see"],
  "excluded_objects": ["objects", "the", "instruction", "says", "to", "leave"],
  "allowed_objects": ["visible", "minus", "excluded"],
  "subtasks": ["put the X on the tray", "put the Y next to the X"]
}

Rules:
- The orange robot arm/kumquat is not an object. Do not include it in visible_objects.
- excluded_objects: any object the instruction says to leave, skip, ignore, or not touch. If none, use [].
- allowed_objects: every object in visible_objects that is NOT in excluded_objects.
- subtasks: one sub-task per allowed object ONLY. Each sub-task is a single pick-and-place action.
- Each sub-task must specify both the object AND its destination (e.g. "put the fork next to the plate", "put the bowl on the tray", "put the plate on the placemat").
- Infer the destination for each object from the instruction and common sense. If the instruction gives a specific destination, use it. If the instruction implies an arrangement (e.g. "set the table"), use spatial language appropriate to the task (e.g. "next to", "on", "in front of").
- Destinations MUST be grounded in the scene. Only use objects or surfaces you can see in the camera observation as destinations.
- If the instruction is vague about destination (e.g. "away", "clean up", "tidy up"), choose the most reasonable visible container or surface (e.g. a tray, bin, or box) as the destination.
- Never use a destination that is not visible in the scene.
- If the instruction uses a category word (e.g. "food", "drinks", "utensils"), identify which visible objects belong to that category and treat them all as excluded (or included).
- EVERY object in allowed_objects MUST have exactly one sub-task. If allowed_objects is not empty, subtasks CANNOT be empty.
- Order subtasks logically. Place base objects before objects that go on top of or relative to them (e.g. place the plate before placing the fork next to the plate).
- visible_objects should include ALL objects and surfaces you see, including containers and destinations (trays, bowls, boxes, plates). An object can be both a destination for one subtask and a moved object in another.

Examples:

Instruction: "put everything on the tray but leave the banana"
{"visible_objects": ["apple", "banana", "orange"], "excluded_objects": ["banana"], "allowed_objects": ["apple", "orange"], "subtasks": ["put the apple on the tray", "put the orange on the tray"]}

Instruction: "clean up the table, don't touch the cup"
{"visible_objects": ["cup", "plate", "fork"], "excluded_objects": ["cup"], "allowed_objects": ["plate", "fork"], "subtasks": ["put the plate on the tray", "put the fork on the tray"]}

Instruction: "put the apple on the tray"
{"visible_objects": ["apple", "banana"], "excluded_objects": [], "allowed_objects": ["apple"], "subtasks": ["put the apple on the tray"]}

Instruction: "put everything away but leave the food"
{"visible_objects": ["banana", "apple", "cup", "book"], "excluded_objects": ["banana", "apple"], "allowed_objects": ["cup", "book"], "subtasks": ["put the cup on the tray", "put the book on the tray"]}

Instruction: "stack the blocks"
{"visible_objects": ["red block", "blue block", "green block"], "excluded_objects": [], "allowed_objects": ["red block", "blue block", "green block"], "subtasks": ["put the blue block on the red block", "put the green block on the blue block"]}
"""

EVALUATION_SYSTEM_PROMPT = """You are a robot task evaluator. You receive a short video clip of a robot arm attempting a sub-task, and the sub-task text. The clip's frames are in time order: earlier frames show the start of the attempt, later frames show the end.

Assess whether the sub-task was completed successfully. Judge the FINAL state (the last frames), but use the motion across the clip as evidence — e.g. whether the object was actually grasped, moved, and released at the destination rather than dropped or knocked aside.

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
        result = json.loads(output.strip())
        if isinstance(result, dict) and "subtasks" in result:
            tasks = result["subtasks"]
            print(f"\nVisible: {result.get('visible_objects')}")
            print(f"Excluded: {result.get('excluded_objects')}")
            print(f"Allowed: {result.get('allowed_objects')}")
        else:
            tasks = result  # fallback for plain array output
        print(f"\nParsed {len(tasks)} sub-tasks:")
        for i, task in enumerate(tasks, 1):
            print(f"  {i}. {task}")
    except json.JSONDecodeError:
        print("\n[WARNING] Output is not valid JSON")    

    return output

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

    interactive_loop(model, processor, camera, image_paths, save_dir)

    if camera is not None:
        camera.stop()


if __name__ == "__main__":
    main()