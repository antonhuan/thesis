"""
Shared utilities for VLA (PI05) embedding extraction and analysis.

Provides model loading, input preparation, VLM hidden-state extraction,
pooling, and image loading — used by both the interactive embedding
comparison tool (vla_embedding_comp.py) and the denoising consistency
test (vla_denoise_consistency.py).
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_policy(checkpoint_path: str, device: str = "cuda", dtype: str = "float32"):
    """Load a PI05Policy from a pretrained checkpoint.

    Sets ``device`` and ``dtype`` on the config *before* loading so weights are
    constructed and mapped directly onto the target device/precision — avoiding
    a redundant second ``.to(device)`` copy. ``dtype="bfloat16"`` roughly halves
    the disk read and host->GPU transfer versus the float32 default.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.pi05 import PI05Policy

    config = PreTrainedConfig.from_pretrained(checkpoint_path)
    config.device = device
    config.dtype = dtype

    policy = PI05Policy.from_pretrained(checkpoint_path, config=config)
    policy = policy.eval()
    return policy


# --------------------------------------------------------------------------- #
# Image loading
# --------------------------------------------------------------------------- #

def load_scene_image(path: str) -> Image.Image:
    """Load a saved scene image from disk, returning a PIL Image in RGB."""
    img = Image.open(path).convert("RGB")
    return img


# --------------------------------------------------------------------------- #
# Input preparation
# --------------------------------------------------------------------------- #

_TOKENIZER_CACHE = {}


def _get_tokenizer(name: str = "google/paligemma-3b-pt-224"):
    """Load (and cache) the PaliGemma tokenizer.

    ``prepare_inputs`` is called once per prompt; without caching the tokenizer
    would be re-downloaded/parsed on every call.
    """
    if name not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer

        _TOKENIZER_CACHE[name] = AutoTokenizer.from_pretrained(name)
    return _TOKENIZER_CACHE[name]


def prepare_inputs(policy, image: Image.Image, instruction: str, device: str = "cuda"):
    """
    Convert a PIL frame + instruction string into the tensors that
    PI05Pytorch.embed_prefix() expects: (images, img_masks, tokens, masks).
    """
    from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch

    config = policy.config

    # --- Image ---
    img = image.convert("RGB")
    img_tensor = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
    img_size = config.image_resolution[0]
    img_tensor = resize_with_pad_torch(img_tensor, img_size, img_size)
    img_tensor = img_tensor.squeeze(0).permute(2, 0, 1).unsqueeze(0).to(device)

    images = [img_tensor]
    img_masks = [torch.ones(1, dtype=torch.bool, device=device)]

    # --- Language tokens ---
    tokenizer = _get_tokenizer()
    encoded = tokenizer(
        instruction,
        return_tensors="pt",
        padding="max_length",
        max_length=config.tokenizer_max_length,
        truncation=True,
    ).to(device)

    tokens = encoded["input_ids"]
    masks = encoded["attention_mask"].bool()

    return images, img_masks, tokens, masks


# --------------------------------------------------------------------------- #
# State-conditioned input preparation (routes through the real preprocessor)
# --------------------------------------------------------------------------- #

def build_preprocessor(checkpoint, config):
    """Load the checkpoint's PI05 *input* pipeline (normalize -> discretize state
    -> "Task: .. , State: .. ; Action: " template -> tokenizer -> device), so the
    prompt the model sees matches training exactly. Returns the preprocessor
    pipeline, or None if it can't be loaded (caller falls back to the bare,
    stateless ``prepare_inputs``)."""
    try:
        from lerobot.policies.factory import make_pre_post_processors

        pre, _ = make_pre_post_processors(config, pretrained_path=str(checkpoint))
        log.info("Loaded checkpoint preprocessor (state-conditioned prompts).")
        return pre
    except Exception as e:  # noqa: BLE001
        log.warning("Could not load checkpoint preprocessor (%s); falling back "
                    "to stateless prompts.", e)
        return None


def prepare_inputs_stateful(policy, preprocessor, image, instruction,
                            initial_pose, device="cuda"):
    """Build (images, img_masks, tokens, masks) for ``embed_prefix`` by routing a
    proper observation — scene image + initial joint pose + task — through the
    checkpoint's real preprocessor. This reproduces the training-time
    ``Task: {instruction}, State: {discretized_pose}; Action:`` prompt (the pose
    is QUANTILES-normalized then binned into 256 levels) and the [0,1]->[-1,1]
    image handling, neither of which the bare ``prepare_inputs`` does.

    ``initial_pose`` is a length-6 real-unit joint vector in motor order
    [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper].
    """
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )

    config = policy.config
    img_keys = list(config.image_features)
    if not img_keys:
        raise ValueError("policy config exposes no image_features")
    cam_key = img_keys[0]

    # Scene image as (C, H, W) float in [0, 1]; the pipeline adds the batch dim
    # and the model's _preprocess_images resizes + maps to [-1, 1].
    img = image.convert("RGB")
    img_t = torch.tensor(np.array(img), dtype=torch.float32) / 255.0  # (H,W,C)
    img_t = img_t.permute(2, 0, 1).contiguous()                       # (C,H,W)

    # Initial pose in real units; the NormalizerProcessorStep normalizes it.
    state = torch.as_tensor(initial_pose, dtype=torch.float32).flatten()

    obs = {
        cam_key: img_t,
        OBS_STATE: state,
        "task": instruction,
    }
    processed = preprocessor(obs)

    images, img_masks = policy._preprocess_images(processed)
    tokens = processed[OBS_LANGUAGE_TOKENS]
    masks = processed[OBS_LANGUAGE_ATTENTION_MASK].bool()
    return images, img_masks, tokens, masks


# --------------------------------------------------------------------------- #
# VLM hidden-state extraction
# --------------------------------------------------------------------------- #

@torch.no_grad()
def extract_vlm_hidden_states(policy, images, img_masks, tokens, masks):
    """
    Run the prefix through the VLM backbone and return the final hidden
    states — the representation that gets cached as KV pairs for the
    action expert.

    Returns:
        hidden_states: [B, seq_len, 2048] — VLM final norm output
        prefix_embs:   [B, seq_len, 2048] — pre-transformer concatenated embeddings
        pad_masks:     [B, seq_len]        — valid token mask
        num_img_tokens: int                — number of image patch tokens (256)
    """
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    model = policy.model

    prefix_embs, pad_masks, att_masks = model.embed_prefix(
        images, img_masks, tokens, masks
    )

    num_img_tokens = prefix_embs.shape[1] - tokens.shape[1]
    vlm = model.paligemma_with_expert.paligemma.model.language_model

    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    att_2d_masks_4d = att_2d_masks[:, None, :, :]
    model_dtype = next(vlm.parameters()).dtype
    att_2d_masks_4d = torch.where(
        att_2d_masks_4d,
        torch.tensor(0.0, dtype=model_dtype, device=att_2d_masks_4d.device),
        torch.tensor(-1e9, dtype=model_dtype, device=att_2d_masks_4d.device),
    )

    vlm_output = vlm.forward(
        inputs_embeds=prefix_embs,
        attention_mask=att_2d_masks_4d,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
    )

    hidden_states = vlm_output.last_hidden_state
    return hidden_states, prefix_embs, pad_masks, num_img_tokens


# --------------------------------------------------------------------------- #
# Pooling utilities
# --------------------------------------------------------------------------- #

def pool_embeddings(hidden_states, pad_masks, method="mean"):
    """Pool sequence-level hidden states into a single vector."""
    if method == "mean":
        mask = pad_masks.unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    elif method == "last":
        lengths = pad_masks.sum(dim=1) - 1
        pooled = hidden_states[torch.arange(len(lengths)), lengths]
    elif method == "cls":
        pooled = hidden_states[:, 0, :]
    else:
        raise ValueError(f"Unknown pooling method: {method}")
    return pooled


def pool_region(hidden_states, pad_masks, start, end, method="mean"):
    """Pool a specific slice of the sequence (e.g. image-only or language-only)."""
    region_hidden = hidden_states[:, start:end, :]
    region_masks = pad_masks[:, start:end]
    return pool_embeddings(region_hidden, region_masks, method)
