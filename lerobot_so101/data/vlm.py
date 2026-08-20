"""
Interactive harness for testing Qwen3-VL task decomposition.

The VLM reasoning layer of the dual-system architecture turns a natural language
instruction + a camera observation into a queue of atomic pick-and-place
sub-tasks. This script is a focused REPL for iterating on that decomposition
alone — load the model once, then type prompts and inspect the sub-tasks.

Decomposition runs as two sequential calls against the same frame (toggle with
/split):
    Pass 1 — identification: image -> list every visible object/surface.
    Pass 2 — decomposition:  image + instruction + that object list -> sub-tasks.

Shared model loading and inference live in vlm_core.py; success evaluation and
the video-clip machinery live in the orchestrator (vlm_robot_orchestrator.py).

Captures frames directly from an Intel RealSense D435 camera (top-down view).

Requirements:
    pip install torch transformers accelerate pillow pyrealsense2 numpy

    # Qwen3-VL requires latest transformers (built from source or >= 4.57.0)
    pip install git+https://github.com/huggingface/transformers

Usage:
    # Live capture from RealSense D435 (default):
    python vlm.py

    # Fall back to static image files if no camera:
    python vlm.py --images top.png

    # Save captured frames to disk:
    python vlm.py --save-frames

    # Skip camera, text-only:
    python vlm.py --no-camera
"""

import argparse
import time
import json
import numpy as np
from pathlib import Path
from PIL import Image

from vlm_core import load_model, generate

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
# System prompts matching the dual-system architecture
# ---------------------------------------------------------------------------

IDENTIFICATION_SYSTEM_PROMPT = """You are a perception module for an orange tabletop robot arm (SO-101, 6-DOF). You receive a camera observation of the workspace. Your ONLY job is to list every object and surface you can see. Pay close attention to objects that might be blocked by other objects and make a reasonable inference as to what the object may be.

You MUST respond in the following JSON format exactly:

{"visible_objects": ["list", "of", "every", "object", "and", "surface"]}

Rules:
- List ALL objects and surfaces on the table, including containers and destinations (trays, bowls, boxes, plates, placemats).
- The orange robot arm/kumquat is part of the robot, NOT an object. Do not include it.
- Use simple, concrete names for what you see (e.g. "cup", "banana", "tray").
- Do not describe the object, use the simplest accurate label. Do not use adjectives such as color, size, texture to describe an object.
- Do NOT plan, decompose, or reason about any instruction. Only identify what is visible.
- If you are unsure of an object's identity, describe it by its most obvious visual feature (colour, shape).
- STACKED AND TOUCHING OBJECTS: an object resting ON TOP of another object is a separate object. List it AND the object beneath it as two entries. Never merge a stack into one label.
- If a shape could plausibly be two touching or overlapping objects rather than one, report it as two. A sudden change in colour, texture, or material across one blob almost always means two objects, not one.

Examples:
{"visible_objects": ["apple", "banana", "orange", "tray"]}
{"visible_objects": ["toy", "pouch", "computer", "tray"]}
A pouch lying on top of a plush toy is two objects, not one:
{"visible_objects": ["banana", "pouch", "toy", "tray", "table"]}
"""

# Second pass of the two-call split: the visible-objects list is provided, so the
# model no longer identifies objects itself — it only reasons about exclusions and
# builds the sub-task queue.
DECOMPOSITION_FROM_OBJECTS_SYSTEM_PROMPT = """You are a task planner for an orange tabletop robot arm (SO-101, 6-DOF). You receive a natural language instruction, a camera observation, and a list of the objects already identified in the scene (visible_objects). Treat that list as ground truth — do NOT add objects to it or remove objects from it.

You MUST respond in the following JSON format exactly:

{
  "visible_objects": ["the", "objects", "you", "were", "given"],
  "excluded_objects": ["objects", "the", "instruction", "says", "to", "leave"],
  "completed_objects": ["objects", "already", "at", "their", "destination"],
  "allowed_objects": ["objects", "to", "move"],
  "subtasks": ["put the X on the tray", "put the Y next to the X"]
}

The four object lists PARTITION visible_objects: every object you were given belongs to
exactly one of excluded_objects, completed_objects, allowed_objects, or the destinations.

Rules:
- visible_objects: echo back exactly the list of objects you were given. Do not change it.
- excluded_objects: any object the instruction says to leave, skip, ignore, or not touch. If none, use [].
- completed_objects: objects that already need no action — they are visibly at the destination the instruction requires (e.g. already on the tray), or they are named in an already-completed list you were given. On a first pass this is normally [].
- allowed_objects: the objects that still need to be moved — visible_objects MINUS excluded, MINUS completed, MINUS destinations.
- The destination container/surface (e.g. the tray, bin, box) is a landmark, not an object to move. Do NOT give it its own sub-task and do NOT list it in allowed_objects, even though it appears in visible_objects.
- subtasks: one sub-task per allowed object ONLY. Each sub-task is a single pick-and-place action.
- Each sub-task must specify both the object AND its destination (e.g. "put the fork next to the plate", "put the bowl on the tray", "put the plate on the placemat").
- Infer the destination for each object from the instruction and common sense. If the instruction gives a specific destination, use it. If the instruction implies an arrangement (e.g. "set the table"), use spatial language appropriate to the task (e.g. "next to", "on", "in front of").
- Destinations MUST come from visible_objects. Never use a destination that is not in visible_objects.
- If the instruction is vague about destination (e.g. "away", "clean up", "tidy up"), choose the most reasonable visible container or surface (e.g. a tray, bin, or box) as the destination.
- If the instruction uses a category word (e.g. "food", "drinks", "utensils"), identify which visible objects belong to that category and treat them all as excluded (or included).
- EVERY object in allowed_objects MUST have exactly one sub-task. If allowed_objects is not empty, subtasks CANNOT be empty.
- NEVER express "there is nothing left to do" by returning an empty subtasks list while allowed_objects is not empty. An object that needs no action belongs in completed_objects (or excluded_objects), not in allowed_objects. subtasks is empty ONLY when allowed_objects is also empty.
- Order subtasks logically. Place base objects before objects that go on top of or relative to them (e.g. place the plate before placing the fork next to the plate).

Examples:

Visible objects: ["apple", "banana", "orange", "tray"]
Instruction: "put everything on the tray but leave the banana"
{"visible_objects": ["apple", "banana", "orange", "tray"], "excluded_objects": ["banana"], "completed_objects": [], "allowed_objects": ["apple", "orange"], "subtasks": ["put the apple on the tray", "put the orange on the tray"]}

Visible objects: ["cup", "plate", "fork", "tray"]
Instruction: "clean up the table, don't touch the cup"
{"visible_objects": ["cup", "plate", "fork", "tray"], "excluded_objects": ["cup"], "completed_objects": [], "allowed_objects": ["plate", "fork"], "subtasks": ["put the plate on the tray", "put the fork on the tray"]}

Visible objects: ["red block", "blue block", "green block"]
Instruction: "stack the blocks"
{"visible_objects": ["red block", "blue block", "green block"], "excluded_objects": [], "completed_objects": [], "allowed_objects": ["red block", "blue block", "green block"], "subtasks": ["put the blue block on the red block", "put the green block on the blue block"]}

A retry pass, where the image shows the plush toy already sitting on the tray but the
pouch still on the table. The plush toy goes in completed_objects — NOT in allowed_objects
with an empty subtask list:
Visible objects: ["banana", "pouch", "plush toy", "tray", "table"]
Instruction: "put everything on the tray except for the banana"
{"visible_objects": ["banana", "pouch", "plush toy", "tray", "table"], "excluded_objects": ["banana"], "completed_objects": ["plush toy"], "allowed_objects": ["pouch"], "subtasks": ["put the pouch on the tray"]}
"""

# Single perception pass used by the final evaluation: name every object AND say
# what it rests on, in one call. It is deliberately never told the instruction,
# the goal, or that a robot acted — the final evaluator was observed filling each
# object's location in from where the instruction *wanted* it rather than from the
# image, and withholding the goal removes the thing it was copying from.
#
# Naming and locating were briefly two calls. They are fused because the join
# between them lost objects: an identification pass reported five objects and the
# localisation pass that followed emitted rows for only two, silently dropping the
# one object whose position actually mattered. With one call the row set IS the
# identification, so there is no gap to fall through.
SCENE_SURVEY_SYSTEM_PROMPT = """You are a perception module for an orange tabletop robot arm (SO-101, 6-DOF). You receive a camera observation of the workspace. Your ONLY job is to list every object you can see and report what each one is currently RESTING ON.

You MUST respond in the following JSON format exactly:

{
  "object_locations": [
    {"object": "<name>", "evidence": "<what you see>", "support": "<what it rests on>"}
  ]
}

What counts as an object:
- Every object on the table gets EXACTLY ONE row. Do not skip an object because it looks unimportant, partly hidden, or hard to name.
- Surfaces and containers — the table, the tray, a bowl, a box — are SUPPORTS, not objects. Name them in "support", but never give them a row of their own.
- The orange robot arm is part of the robot, not an object. Do not give it a row. Ignore any text or label printed on it; that is not an object either.
- Pay close attention to objects partly blocked by other objects. A blocked object still gets a row — infer what it is from the part you can see.
- STACKED AND TOUCHING OBJECTS: an object resting ON TOP of another object is a separate object. Give it a row AND give the object beneath it a row. Never merge a stack into one entry.
- If a shape could plausibly be two touching or overlapping objects rather than one, report it as two. A sudden change in colour, texture, or material across one blob almost always means two objects, not one.
- Use simple, concrete names for what you see (e.g. "cup", "banana", "pouch"). If you are unsure what something is, name it by its most obvious visual feature (colour or shape).

How to report where it is:
- Fill in "evidence" BEFORE "support". Describe where the object actually sits in this image relative to the tray — inside its rim, touching its edge, or clear of it on the bare table, and roughly which side. Then, and only then, name the support.
- An object is on the tray ONLY if it lies inside the tray's rim. An object that is beside the tray, in front of it, behind it, or merely near it is on the table, not on the tray.
- If an object rests on top of another object rather than directly on a surface, name that object as its support.
- Report what you SEE. Do not guess where an object ought to be, and do not assume anything has been moved, tidied, or put away.

Examples:
{"object_locations": [{"object": "apple", "evidence": "sitting inside the tray's rim, in the middle of the tray", "support": "tray"}, {"object": "banana", "evidence": "on the bare table to the left of the tray, not touching it", "support": "table"}]}
{"object_locations": [{"object": "pouch", "evidence": "lying on the bare table well to the left of the tray, clear of its rim", "support": "table"}, {"object": "plush toy", "evidence": "lying inside the tray, its body within the rim", "support": "tray"}]}
A plush toy lying on top of a pouch is two objects, and the pouch is what the toy rests on:
{"object_locations": [{"object": "plush toy", "evidence": "lying across the middle of the table, on top of the brown pouch, clear of the tray", "support": "pouch"}, {"object": "pouch", "evidence": "under the plush toy on the bare table, well left of the tray", "support": "table"}, {"object": "banana", "evidence": "on the bare table at the far left, not touching the tray", "support": "table"}]}
"""

# Single-call decomposition (identify + plan in one shot). Kept as the /split
# toggle-off path for A/B comparison against the two-pass split above.
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


# ---------------------------------------------------------------------------
# Vocabulary refinement prompts (used by the orchestrator's closed-loop
# failure-classification path)
# ---------------------------------------------------------------------------

REFINEMENT_SYSTEM_PROMPT = """You previously generated a robot instruction that the robot failed to execute. The failure suggests the object name in the instruction confused the robot.

Given the failed instruction and a description of what went wrong, suggest 3 alternative names for the target object. Use simple, common, single-word nouns when possible. Return ONLY the three alternatives, one per line, most likely first. No numbering, no explanation."""

GUIDED_VOCAB_MATCH_SYSTEM_PROMPT = """You identified an object using an open-vocabulary label, but the robot could not execute instructions with that name or the alternatives you suggested.

The robot was trained with a fixed set of object labels. Given the label you used and the training vocabulary, determine which training label refers to the same object. If none match, say NONE.

Return ONLY the matching training label (exactly as it appears in the list) or the word NONE."""


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
# Two-pass decomposition (identify -> decompose)
# ---------------------------------------------------------------------------

def parse_visible_objects(output: str) -> list[str]:
    """Extract the visible_objects list from an identification-pass output.

    Tolerates either the documented dict schema or a bare JSON array. Returns []
    if nothing parseable is found (the caller can still proceed to decomposition
    with an empty list).
    """
    try:
        result = json.loads(output.strip())
    except json.JSONDecodeError:
        return []
    if isinstance(result, dict):
        objs = result.get("visible_objects", [])
    elif isinstance(result, list):
        objs = result
    else:
        objs = []
    return [str(o) for o in objs] if isinstance(objs, list) else []


def identify_objects(model, processor, image_content, temperature: float = 0.7) -> list[str]:
    """Pass 1 — image -> list of visible objects/surfaces.

    `image_content` is a prebuilt content list (image + label dicts). Returns the
    parsed object list; [] on parse failure.
    """
    user_content = list(image_content) if image_content else []
    user_content.append(
        {"type": "text", "text": "\nList every visible object and surface."}
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": IDENTIFICATION_SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]
    output = generate(model, processor, messages, temperature=temperature)
    print(f"\nIdentification output:\n{output}")
    objects = parse_visible_objects(output)
    if not objects:
        print("[WARNING] Identification pass produced no objects (parse failed or empty).")
    return objects


def parse_object_locations(output: str) -> list[dict]:
    """Extract the object_locations rows from a localisation-pass output.

    Tolerates either the documented dict schema or a bare JSON array of rows, and
    keeps only well-formed rows (a dict naming an object). Returns [] if nothing
    parseable is found, so the caller can fall back to a single-call evaluation
    rather than judging against garbage.
    """
    text = output.strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return []
        try:
            result = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []
    if isinstance(result, dict):
        rows = result.get("object_locations", [])
    elif isinstance(result, list):
        rows = result
    else:
        rows = []
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict) and r.get("object")]


def survey_scene(model, processor, image_content,
                 temperature: float = 0.7) -> list[dict]:
    """Perception pass — image -> every object and what it rests on.

    `image_content` is a prebuilt content list (image + label dicts), as for
    `identify_objects`. Returns the parsed rows; [] on parse failure.

    Takes no object list and no instruction: naming and locating are one call so
    nothing can be lost between them, and the goal is withheld so the model
    cannot report where an object was *supposed* to end up. See
    SCENE_SURVEY_SYSTEM_PROMPT.
    """
    user_content = list(image_content) if image_content else []
    user_content.append({
        "type": "text",
        "text": "\nList every object you can see and what each one is resting on.",
    })
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SCENE_SURVEY_SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]
    output = generate(model, processor, messages, temperature=temperature)
    print(f"\nScene survey output:\n{output}")
    rows = parse_object_locations(output)
    if not rows:
        print("[WARNING] Scene survey produced no rows (parse failed or empty).")
    return rows


def decompose_from_objects(model, processor, image_content, prompt: str,
                           visible_objects: list[str],
                           temperature: float = 0.7,
                           extra_context: str = "") -> str:
    """Pass 2 — image + instruction + given object list -> decomposition JSON (raw).

    `extra_context` is appended after the instruction (e.g. why a previous
    attempt at this same instruction was judged incomplete).
    """
    text = (
        f"\nVisible objects: {json.dumps(visible_objects)}"
        f"\nInstruction: {prompt}"
    )
    if extra_context:
        text += f"\n\n{extra_context}"
    user_content = list(image_content) if image_content else []
    user_content.append({"type": "text", "text": text})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": DECOMPOSITION_FROM_OBJECTS_SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]
    return generate(model, processor, messages, temperature=temperature)


# ---------------------------------------------------------------------------
# Test routine
# ---------------------------------------------------------------------------

def test_decomposition(model, processor, prompt: str,
                       camera=None, image_paths=None, save_dir=None,
                       two_pass: bool = True, temperature: float = 0.7,
                       extra_context: str = ""):
    """Test task decomposition with camera or image files.

    Two-pass (default) runs identification then decomposition against the SAME
    captured frame. Single-call (two_pass=False) uses the combined
    DECOMPOSITION_SYSTEM_PROMPT for A/B comparison.

    `extra_context` mirrors the orchestrator's re-decompose argument, so a retry
    round can be replayed offline against a saved frame (see /context).
    """
    image_content, img_desc = get_image_content(camera, image_paths, save_dir)

    print(f"\n{'='*60}")
    print(f"TASK DECOMPOSITION TEST ({'two-pass' if two_pass else 'single-call'})")
    print(f"Prompt: \"{prompt}\"")
    print(f"Image source: {img_desc or 'text-only'}")
    if extra_context:
        print(f"Extra context:\n{extra_context}")
    print(f"{'='*60}")

    # Text-only fallback: a described scene when there is no image source.
    if image_content is None:
        image_content = [{
            "type": "text",
            "text": (
                "The robot is at a tabletop with an orange, a blue cup, "
                "a red cup, and a plate."
            ),
        }]

    if two_pass:
        visible = identify_objects(model, processor, image_content, temperature)
        print(f"\nPass 1 — identified objects ({len(visible)}): {visible}")
        output = decompose_from_objects(
            model, processor, image_content, prompt, visible, temperature,
            extra_context=extra_context,
        )
    else:
        text = f"\nInstruction: {prompt}"
        if extra_context:
            text += f"\n\n{extra_context}"
        user_content = list(image_content)
        user_content.append({"type": "text", "text": text})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": DECOMPOSITION_SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
        ]
        output = generate(model, processor, messages, temperature=temperature)

    print(f"\nModel output:\n{output}")

    # Try to parse as JSON
    try:
        result = json.loads(output.strip())
        if isinstance(result, dict) and "subtasks" in result:
            tasks = result["subtasks"]
            print(f"\nVisible: {result.get('visible_objects')}")
            print(f"Excluded: {result.get('excluded_objects')}")
            print(f"Completed: {result.get('completed_objects')}")
            print(f"Allowed: {result.get('allowed_objects')}")
            # Mirrors the orchestrator's coverage check: allowed_objects is the
            # work that remains, so anything in it with no sub-task is a bug.
            joined = " ".join(str(t) for t in tasks).lower()
            missing = [str(o) for o in (result.get("allowed_objects") or [])
                       if str(o).lower() not in joined]
            if missing:
                print(f"[WARNING] allowed_objects with no sub-task: {missing}")
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
  /split              Toggle two-pass (identify -> decompose) vs single call
                      (current: {split})
  /temp <value>       Set temperature (current: {temp})
  /save               Toggle saving frames to disk (current: {save})
  /context            Enter retry context to append to every prompt, so a
                      re-decompose round can be replayed against a saved frame.
                      Ends on a blank line; /context with none set clears it.
                      (current: {context})
  /help               Show this help
  /quit               Exit
""".strip()


def interactive_loop(model, processor, camera, image_paths, save_dir):
    """Interactive REPL — model stays loaded, type prompts freely."""

    temp = 0.1
    saving = save_dir is not None
    two_pass = True
    extra_context = ""  # set via /context to replay a re-decompose round

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
                split="two-pass" if two_pass else "single-call",
                context=f"{len(extra_context)} chars" if extra_context else "none",
            ))

        elif user_input == "/context":
            if extra_context:
                extra_context = ""
                print("Retry context cleared.")
            else:
                print("Paste the retry context, then a blank line to finish:")
                lines = []
                while True:
                    try:
                        line = input()
                    except (EOFError, KeyboardInterrupt):
                        break
                    if not line.strip():
                        break
                    lines.append(line)
                extra_context = "\n".join(lines)
                print(f"Retry context set ({len(extra_context)} chars)."
                      if extra_context else "Retry context left empty.")

        elif user_input == "/split":
            two_pass = not two_pass
            print(f"Decomposition mode: {'two-pass (identify -> decompose)' if two_pass else 'single-call'}")

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
                two_pass=two_pass, temperature=temp,
                extra_context=extra_context,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive harness for testing Qwen3-VL task decomposition"
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
