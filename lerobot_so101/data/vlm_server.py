"""
VLM inference server — runs on the 3080 (or any GPU with enough VRAM).

Exposes two endpoints:
  POST /decompose   — image + prompt -> sub-task list
  POST /evaluate    — image + sub-task -> success judgement

Start:
  pip install fastapi uvicorn pillow torch transformers accelerate
  python vlm_server.py --host 0.0.0.0 --port 9090
"""

import argparse
import base64
import io
import json
import logging
import re
import time

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image

# ---------------------------------------------------------------------------
# System prompts (keep in sync with vlm.py)
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
# Model loading and inference (same as vlm.py)
# ---------------------------------------------------------------------------

def load_model(model_name: str):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    logging.info(f"Loading {model_name}...")
    t0 = time.time()

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    logging.info(f"Model loaded in {time.time() - t0:.1f}s")
    logging.info(f"Device: {model.device}")
    logging.info(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    return model, processor


def generate(model, processor, messages, max_new_tokens=1024, temperature=0.7):
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
    logging.info(f"Generated {n_tokens} tokens in {elapsed:.1f}s "
                 f"({n_tokens / elapsed:.1f} tok/s)")

    return output_text


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _strip_think_tags(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_subtask_list(output: str) -> list[str]:
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
# Image helpers
# ---------------------------------------------------------------------------

def decode_image(b64_string: str) -> Image.Image:
    """Decode a base64-encoded image to PIL Image."""
    image_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_bytes))


def build_user_content(frame: Image.Image | None, text: str) -> list[dict]:
    content = []
    if frame is not None:
        content.append({"type": "text", "text": "[top-down camera]"})
        content.append({"type": "image", "image": frame})
    content.append({"type": "text", "text": text})
    return content


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="VLM Inference Server")

# These get populated in main() before uvicorn starts
_model = None
_processor = None


class DecomposeRequest(BaseModel):
    prompt: str
    image_b64: str | None = None  # base64-encoded PNG/JPEG
    temperature: float = 0.7


class DecomposeResponse(BaseModel):
    subtasks: list[str]
    raw_output: str


class EvaluateRequest(BaseModel):
    sub_task: str
    image_b64: str | None = None
    temperature: float = 0.7


class EvaluateResponse(BaseModel):
    success: bool
    reason: str
    raw_output: str


@app.post("/decompose", response_model=DecomposeResponse)
def decompose(req: DecomposeRequest):
    frame = decode_image(req.image_b64) if req.image_b64 else None

    text = f"\nInstruction: {req.prompt}"
    if frame is None:
        text = (
            "No camera observation is available. Decompose the instruction "
            "based on the text alone.\n" + text
        )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": DECOMPOSITION_SYSTEM_PROMPT}]},
        {"role": "user", "content": build_user_content(frame, text)},
    ]

    output = generate(_model, _processor, messages, temperature=req.temperature)
    logging.info(f"Raw decomposition output: {output!r}")

    try:
        subtasks = parse_subtask_list(output)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return DecomposeResponse(subtasks=subtasks, raw_output=output)


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    frame = decode_image(req.image_b64) if req.image_b64 else None

    text = (
        f'\nThe robot just attempted this sub-task: "{req.sub_task}"\n'
        "Did it succeed?"
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": EVALUATION_SYSTEM_PROMPT}]},
        {"role": "user", "content": build_user_content(frame, text)},
    ]

    output = generate(_model, _processor, messages, temperature=req.temperature)
    logging.info(f"Raw evaluation output: {output!r}")

    try:
        result = parse_evaluation(output)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return EvaluateResponse(
        success=bool(result.get("success")),
        reason=result.get("reason", ""),
        raw_output=output,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": _model_name}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_model_name = ""


def main():
    global _model, _processor, _model_name

    parser = argparse.ArgumentParser(description="VLM inference server")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct",
                        help="HuggingFace model name or path")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=9090,
                        help="Port (default: 9090)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    _model_name = args.model
    _model, _processor = load_model(args.model)

    logging.info(f"Starting VLM server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()