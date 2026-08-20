"""
Shared utilities for VLA (PI05) embedding extraction and analysis.

Provides model loading, input preparation, VLM hidden-state extraction,
pooling, and image loading — used by both the interactive embedding
comparison tool (vla_embedding_comp.py) and the denoising consistency
test (vla_denoise_consistency.py).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_policy(checkpoint_path: str, device: str = "cuda"):
    """Load a PI05Policy from a pretrained checkpoint."""
    from lerobot.policies.pi05 import PI05Policy

    policy = PI05Policy.from_pretrained(checkpoint_path)
    policy = policy.to(device).eval()
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

def prepare_inputs(policy, image: Image.Image, instruction: str, device: str = "cuda"):
    """
    Convert a PIL frame + instruction string into the tensors that
    PI05Pytorch.embed_prefix() expects: (images, img_masks, tokens, masks).
    """
    from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch
    from transformers import AutoTokenizer

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
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
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
