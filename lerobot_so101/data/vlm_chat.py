"""
Bare interactive chat with Qwen3-VL over the RealSense camera.

A stripped clone of vlm.py with all of the dual-system scaffolding removed: no
system prompt, no identification pass, no decomposition schema, no JSON parsing.
Whatever you type goes to the model verbatim alongside a freshly captured
top-down frame, and the raw reply is printed. Useful for sanity-checking what
the VLM actually sees before blaming the task prompts.

Model loading and inference are reused from vlm_core.py; the RealSense wrapper
is reused from vlm.py.

Requirements:
    pip install torch transformers accelerate pillow pyrealsense2 numpy

    # Qwen3-VL requires latest transformers (built from source or >= 4.57.0)
    pip install git+https://github.com/huggingface/transformers

Usage:
    # Live capture from RealSense D435 (default), one fresh frame per message:
    python vlm_chat.py

    # Static image files instead of the camera:
    python vlm_chat.py --images top.png

    # Save every captured frame to ./frames/:
    python vlm_chat.py --save-frames

    # Text-only, no image at all:
    python vlm_chat.py --no-camera
"""

import argparse
from pathlib import Path

from vlm import RealSenseCamera, capture_scene, build_image_content_paths
from vlm_core import load_model, generate


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

INTERACTIVE_HELP = """
Commands:
  <any text>          Send to the model with the current camera frame
  /shot               Capture a frame now and show it as the frozen frame
  /freeze             Toggle reusing one frame vs capturing per message
                      (current: {freeze})
  /multi              Toggle multi-turn history vs stateless single turns
                      (current: {multi})
  /clear              Clear the conversation history
  /temp <value>       Set temperature (current: {temp})
  /tokens <n>         Set max new tokens (current: {tokens})
  /save               Toggle saving frames to disk (current: {save})
  /help               Show this help
  /quit               Exit
""".strip()


def interactive_loop(model, processor, camera, image_paths, save_dir):
    """Interactive REPL — model stays loaded, type anything you like."""

    temp = 0.1
    max_new_tokens = 2048
    saving = save_dir is not None
    freeze = False
    multi_turn = False
    frozen_frame = None
    history: list[dict] = []

    def image_content():
        """Content list for this turn: frozen frame, live capture, or files."""
        nonlocal frozen_frame

        if camera is not None:
            if freeze and frozen_frame is not None:
                frame = frozen_frame
            else:
                frame = capture_scene(camera, save_dir=save_dir if saving else None)
                frozen_frame = frame
            return [{"type": "image", "image": frame}]
        if image_paths:
            return build_image_content_paths(image_paths)
        return []

    print(f"\n{'='*60}")
    print("CHAT MODE — no system prompt, no task scaffolding.")
    print(f"Image source: {'RealSense' if camera else (image_paths or 'text-only')}")
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
        if user_input in ("/quit", "/exit"):
            print("Exiting.")
            break

        elif user_input == "/help":
            print(INTERACTIVE_HELP.format(
                temp=temp,
                tokens=max_new_tokens,
                save="ON" if saving else "OFF",
                freeze="frozen" if freeze else "fresh capture per message",
                multi="ON" if multi_turn else "OFF (stateless)",
            ))

        elif user_input == "/shot":
            if camera is None:
                print("No camera — nothing to capture.")
            else:
                frozen_frame = capture_scene(
                    camera, save_dir=save_dir if saving else None
                )
                print(f"Captured frame: {frozen_frame.size[0]}x{frozen_frame.size[1]}")

        elif user_input == "/freeze":
            freeze = not freeze
            if freeze and frozen_frame is None and camera is not None:
                frozen_frame = capture_scene(
                    camera, save_dir=save_dir if saving else None
                )
            print(f"Frame mode: {'frozen (reusing last frame)' if freeze else 'fresh capture per message'}")

        elif user_input == "/multi":
            multi_turn = not multi_turn
            print(f"Multi-turn history: {'ON' if multi_turn else 'OFF (stateless)'}")

        elif user_input == "/clear":
            history = []
            print("History cleared.")

        elif user_input.startswith("/temp "):
            try:
                temp = float(user_input[6:].strip())
                print(f"Temperature set to {temp}")
            except ValueError:
                print("Usage: /temp <float>  (e.g. /temp 0.3)")

        elif user_input.startswith("/tokens "):
            try:
                max_new_tokens = int(user_input[8:].strip())
                print(f"Max new tokens set to {max_new_tokens}")
            except ValueError:
                print("Usage: /tokens <int>  (e.g. /tokens 512)")

        elif user_input == "/save":
            saving = not saving
            if saving and save_dir is None:
                save_dir = Path("./frames")
            print(f"Frame saving: {'ON — saving to ./frames/' if saving else 'OFF'}")

        elif user_input.startswith("/"):
            print(f"Unknown command: {user_input}. Type /help for options.")

        # --- Plain message to the model ---
        else:
            content = image_content()
            content.append({"type": "text", "text": user_input})

            messages = (history if multi_turn else []) + [
                {"role": "user", "content": content}
            ]

            output = generate(
                model, processor, messages,
                max_new_tokens=max_new_tokens,
                temperature=temp,
            )
            print(f"\n{output}")

            if multi_turn:
                history = messages + [
                    {"role": "assistant", "content": [{"type": "text", "text": output}]}
                ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bare chat REPL for Qwen3-VL with RealSense camera input"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-VL-4B-Instruct",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--images", nargs="*", default=None,
        help="Paths to image files (overrides live camera)",
    )
    parser.add_argument(
        "--no-camera", action="store_true",
        help="Disable RealSense camera, chat text-only",
    )
    parser.add_argument(
        "--save-frames", action="store_true",
        help="Save captured camera frames to ./frames/",
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
