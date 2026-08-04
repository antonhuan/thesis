"""
Shared Qwen3-VL inference infrastructure.

Model loading and generation used by both the decomposition-testing REPL
(vlm.py) and the dual-system orchestrator (vlm_robot_orchestrator.py). Kept
separate so vlm.py can stay a focused task-decomposition harness while the
orchestrator reuses the same (video-capable) inference path for evaluation.

Requirements:
    pip install torch transformers accelerate pillow numpy

    # Qwen3-VL requires latest transformers (built from source or >= 4.57.0)
    pip install git+https://github.com/huggingface/transformers
"""

import time

import torch


# ---------------------------------------------------------------------------
# Model loading
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


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

# Set to False the first time the installed processor rejects the video_metadata
# kwarg, so the unsupported path is probed once per process instead of per call.
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


def generate(model, processor, messages: list, max_new_tokens: int = 2048,
             temperature: float = 0.7) -> str:
    """Run inference and return generated text.

    Uses qwen_vl_utils.process_vision_info so both image and video message content
    work. For image-only messages (decompose/replan) there are no videos, so the
    behaviour is identical to a plain image request.

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
        # Nothing pins transformers, so whether this version's processor accepts
        # video_metadata depends on the image build. Retry once without it and
        # remember, so the probe costs one failed call per process rather than one
        # per evaluation. Without this the orchestrator would see a hard error on
        # every single judgement.
        if "video_metadata" not in video_kwargs:
            raise
        _VIDEO_METADATA_SUPPORTED = False
        print(f"[WARNING] Processor rejected video_metadata ({e}) — retrying without "
              "it. Frame timestamps will fall back to the model's default frame rate.")
        video_kwargs.pop("video_metadata")
        inputs = _call_processor(video_kwargs)

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
