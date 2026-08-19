"""
Shared VLM inference infrastructure.

Model loading and generation used by both the decomposition-testing REPL
(vlm.py) and the dual-system orchestrator (vlm_robot_orchestrator.py). Kept
separate so vlm.py can stay a focused task-decomposition harness while the
orchestrator reuses the same (video-capable) inference path for evaluation.

Supports Qwen VL models (Qwen3-VL, Qwen2.5-VL, Qwen2-VL) via qwen_vl_utils,
and other VLMs (Gemma 3/4, InternVL3, etc.) via the standard
AutoModelForImageTextToText interface.

Requirements:
    pip install torch transformers accelerate pillow numpy

    # For Qwen VL models (default):
    pip install qwen-vl-utils
    pip install git+https://github.com/huggingface/transformers  # >= 4.57.0

    # For Gemma 3/4, InternVL3, etc.:
    pip install transformers>=4.49.0
"""

import time

import torch


# ---------------------------------------------------------------------------
# Model detection
# ---------------------------------------------------------------------------

def _is_qwen_vl(model) -> bool:
    """Check whether a loaded model uses a Qwen VL architecture.

    Qwen VL models use qwen_vl_utils for image/video preprocessing in the
    generate path. All other models use the generic apply_chat_template path.
    """
    model_type = getattr(getattr(model, "config", None), "model_type", "")
    model_type = model_type.lower()
    return "qwen" in model_type and "vl" in model_type


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str = "Qwen/Qwen3-VL-4B-Instruct"):
    """Load model and processor.

    Qwen VL models are loaded with their version-specific model class for
    reliable behaviour. All other models use AutoModelForImageTextToText,
    which dispatches to the correct architecture from the checkpoint config.
    """
    from transformers import AutoProcessor

    print(f"Loading {model_name}...")
    t0 = time.time()

    name_lower = model_name.lower()

    # --- Resolve model class ------------------------------------------------
    # Qwen VL: use the explicit class (known reliable, required for
    # qwen_vl_utils compatibility).  Fall back to AutoModel if the
    # version-specific class isn't available in the installed transformers.
    ModelClass = None

    if "qwen" in name_lower and "vl" in name_lower:
        if "qwen3" in name_lower:
            try:
                from transformers import Qwen3VLForConditionalGeneration
                ModelClass = Qwen3VLForConditionalGeneration
            except ImportError:
                pass
        elif "qwen2.5" in name_lower or "qwen2_5" in name_lower:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                ModelClass = Qwen2_5_VLForConditionalGeneration
            except ImportError:
                pass
        elif "qwen2" in name_lower:
            try:
                from transformers import Qwen2VLForConditionalGeneration
                ModelClass = Qwen2VLForConditionalGeneration
            except ImportError:
                pass

    if ModelClass is None:
        from transformers import AutoModelForImageTextToText
        ModelClass = AutoModelForImageTextToText

    # --- Load ---------------------------------------------------------------
    model = ModelClass.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,   # needed for InternVL, MiniCPM-V, etc.
    )
    # Quantised alternative (uncomment to fit alongside the policy server):
    # from transformers import BitsAndBytesConfig
    # model = ModelClass.from_pretrained(
    #     model_name,
    #     quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    #     device_map="auto",
    #     trust_remote_code=True,
    # )

    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    print(f"Model loaded in {time.time() - t0:.1f}s")
    try:
        print(f"Device: {model.device}")
    except Exception:
        print("Device: distributed (device_map='auto')")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    return model, processor


# ---------------------------------------------------------------------------
# Shared generation backend
# ---------------------------------------------------------------------------

def _run_generation(model, processor, inputs, max_new_tokens: int,
                    temperature: float) -> str:
    """Run model.generate on prepared inputs and decode the output.

    Shared by both the Qwen and generic inference paths — everything
    downstream of input tokenisation is model-agnostic.
    """
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
# Qwen VL inference (image + video via qwen_vl_utils)
# ---------------------------------------------------------------------------

# Probed once per process; set to False if the processor rejects video_metadata.
_VIDEO_METADATA_SUPPORTED = True


def build_video_metadata(video_inputs: list, fps) -> list | None:
    """Build one metadata entry per video so Qwen3-VL can construct timestamps.

    Qwen3-VL derives frame timestamps from `video_metadata`, not from the `fps`
    kwarg. We pass pre-sampled frames (a list of PIL images), so nothing
    upstream supplies metadata and the model falls back to assuming fps=24 —
    which mislabels a 20s attempt as under a second.

    Returns None if metadata cannot be built for the installed transformers
    version; the caller then omits it and we are no worse off than before. Every
    such path warns — falling back silently is indistinguishable in the logs from
    the fixed path, which is how the fps=24 bug hid in the first place.
    """
    if not fps or fps <= 0:
        print(f"[WARNING] No usable clip fps ({fps!r}) — skipping video metadata; "
              "the model will assume its default frame rate.")
        return None

    try:
        from transformers.video_utils import VideoMetadata
    except Exception:
        print("[WARNING] transformers.video_utils.VideoMetadata unavailable — "
              "skipping video metadata; the model will assume its default frame rate.")
        return None

    metadata = []
    for video in video_inputs:
        try:
            n = len(video)
        except TypeError:
            print("[WARNING] Video input has no length — skipping video metadata; "
                  "the model will assume its default frame rate.")
            return None
        fields = {
            "fps": fps,
            "total_num_frames": n,
            # n frames at `fps` span (n - 1) intervals, not n.
            "duration": (n - 1) / fps,
            "frames_indices": list(range(n)),
            "video_backend": "pil",
        }
        try:
            # Only pass what this version's dataclass actually declares.
            import dataclasses
            declared = {f.name for f in dataclasses.fields(VideoMetadata)}
            metadata.append(
                VideoMetadata(**{k: v for k, v in fields.items() if k in declared})
            )
        except Exception as e:
            print(f"[WARNING] Could not build VideoMetadata ({e}) — skipping video "
                  "metadata; the model will assume its default frame rate.")
            return None

    return metadata


def _generate_qwen(model, processor, messages: list, max_new_tokens: int,
                   temperature: float) -> str:
    """Qwen VL inference path.

    Uses qwen_vl_utils.process_vision_info so both image and video message
    content work.  For image-only messages (decompose/replan) there are no
    videos, so the behaviour is identical to a plain image request.

    process_vision_info returns fps as one entry per video ([] with no videos,
    [2.0] with one), but the Qwen3-VL processor declares fps as a strict
    int | float | None — so it is normalised to a scalar before the call.
    """
    global _VIDEO_METADATA_SUPPORTED

    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    video_kwargs = dict(video_kwargs or {})
    if not video_inputs:
        # No videos: fps comes back as [], which fails the strict type check.
        video_kwargs = {}
    elif isinstance(video_kwargs.get("fps"), (list, tuple)):
        # One fps per video; every call site here sends exactly one.
        fps = video_kwargs["fps"]
        video_kwargs["fps"] = fps[0] if fps else None

    if (video_inputs and "video_metadata" not in video_kwargs
            and _VIDEO_METADATA_SUPPORTED):
        metadata = build_video_metadata(video_inputs, video_kwargs.get("fps"))
        if metadata is not None:
            video_kwargs["video_metadata"] = metadata

    def _call_processor(kwargs):
        return processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            **kwargs,
        )

    try:
        inputs = _call_processor(video_kwargs)
    except (TypeError, ValueError) as e:
        # Whether this version's processor accepts video_metadata depends on
        # the transformers build. Retry once without it and remember, so the
        # probe costs one failed call per process rather than one per eval.
        if "video_metadata" not in video_kwargs:
            raise
        _VIDEO_METADATA_SUPPORTED = False
        print(f"[WARNING] Processor rejected video_metadata ({e}) — retrying without "
              "it. Frame timestamps will fall back to the model's default frame rate.")
        video_kwargs.pop("video_metadata")
        inputs = _call_processor(video_kwargs)

    return _run_generation(model, processor, inputs, max_new_tokens, temperature)


# ---------------------------------------------------------------------------
# Generic VLM inference (Gemma 3/4, InternVL3, etc.)
# ---------------------------------------------------------------------------

def _generate_generic(model, processor, messages: list, max_new_tokens: int,
                      temperature: float) -> str:
    """Generic inference path for non-Qwen VLMs.

    Uses processor.apply_chat_template(tokenize=True) which handles image
    extraction and tokenisation in a single call.  This works for any VLM
    whose HuggingFace processor follows the standard multimodal chat template
    interface (Gemma 3/4, InternVL3, Phi-4-multimodal, etc.).

    Video content is NOT supported on this path — use a Qwen VL model for
    video-based success evaluation in the orchestrator.
    """
    # Warn on video content (only supported via qwen_vl_utils)
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "video":
                    print("[WARNING] Video content found but non-Qwen VLM in use. "
                          "Video-based success evaluation requires a Qwen VL model "
                          "with qwen_vl_utils installed. Video input will be skipped.")
                    break

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    return _run_generation(model, processor, inputs, max_new_tokens, temperature)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def generate(model, processor, messages: list, max_new_tokens: int = 2048,
             temperature: float = 0.7) -> str:
    """Run inference and return generated text.

    Dispatches to Qwen-specific or generic inference path based on the loaded
    model's architecture:

    - Qwen VL → qwen_vl_utils for image AND video preprocessing (full
      orchestrator support including video-based success evaluation).
    - Everything else → processor.apply_chat_template(tokenize=True) for
      image support.  Video evaluation is not available on this path.

    The public signature is unchanged — callers (vlm.py, orchestrator) do not
    need to know which backend is active.
    """
    if _is_qwen_vl(model):
        return _generate_qwen(model, processor, messages, max_new_tokens, temperature)
    else:
        return _generate_generic(model, processor, messages, max_new_tokens, temperature)