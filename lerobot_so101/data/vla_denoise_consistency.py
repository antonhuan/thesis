"""
Denoising consistency test for PI0.5 vocabulary sensitivity.

For each prompt, runs the VLA's flow-matching denoiser multiple times with
different noise seeds on the same scene image.  A prompt the VLA recognises
produces a tight cluster of trajectories (strong attractor); an unrecognised
prompt yields high-variance outputs (no attractor).  The per-prompt action
standard deviation is the confidence signal.

Usage:
    # With a saved image:
    python vla_denoise_consistency.py \
        --checkpoint /path/to/finetuned/checkpoint \
        --image_path scene.png

    # With a live RealSense capture:
    python vla_denoise_consistency.py \
        --checkpoint /path/to/finetuned/checkpoint

    # Custom seeds / steps:
    python vla_denoise_consistency.py \
        --checkpoint /path/to/finetuned/checkpoint \
        --image_path scene.png \
        --n_seeds 20 --num_steps 10
"""

import argparse
import logging
import time
from collections import OrderedDict

import torch

from vla_embedding_utils import load_policy, load_scene_image, prepare_inputs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Prompt sets
# --------------------------------------------------------------------------- #

PROMPTS = OrderedDict([
    # Training labels (should be recognised)
    ("banana",           "put the banana on the tray"),
    ("toy",              "put the toy on the tray"),
    ("pouch",            "put the pouch on the tray"),
    # Synonyms (may or may not be recognised)
    ("purse",            "put the purse on the tray"),
    ("bag",              "put the bag on the tray"),
    ("plush toy",        "put the plush toy on the tray"),
    ("stuffed animal",   "put the stuffed animal on the tray"),
    ("fruit",            "put the fruit on the tray"),
    ("yellow banana",    "put the yellow banana on the tray"),
    ("drawstring pouch", "put the drawstring pouch on the tray"),
    ("brown cylinder",     "put the brown cylinder on the tray"),
    # Out-of-domain (should be unrecognised)
    ("mug",              "put the mug on the tray"),
    ("laptop",           "put the laptop on the tray"),
    ("shoe",             "put the shoe on the tray"),
])

TRAINING_LABELS = {"banana", "toy", "pouch"}


# --------------------------------------------------------------------------- #
# Core: cached prefix + denoising loop
# --------------------------------------------------------------------------- #

@torch.no_grad()
def encode_prefix(model, images, img_masks, tokens, masks):
    """
    Run the VLM prefix (image + language) once and return the cached KV
    pairs plus the prefix padding masks needed by denoise_step.
    """
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, tokens, masks
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)

    model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"

    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return prefix_pad_masks, past_key_values


@torch.no_grad()
def denoise_with_cached_kv(model, prefix_pad_masks, past_key_values,
                           noise, num_steps=10):
    """
    Run the flow-matching denoising loop using a pre-computed KV cache.
    denoise_step() deep-copies past_key_values internally, so the cache
    is safe to reuse across calls.
    """
    device = noise.device
    bsize = noise.shape[0]
    dt = -1.0 / num_steps
    x_t = noise.clone()

    for step in range(num_steps):
        t = 1.0 + step * dt
        time_tensor = torch.tensor(t, dtype=torch.float32,
                                   device=device).expand(bsize)
        v_t = model.denoise_step(prefix_pad_masks, past_key_values,
                                 x_t, time_tensor)
        x_t = x_t + dt * v_t

    return x_t


# --------------------------------------------------------------------------- #
# Multi-seed consistency measurement
# --------------------------------------------------------------------------- #

def measure_consistency(policy, image, instruction, n_seeds=10,
                        num_steps=10, device="cuda"):
    """
    Run *n_seeds* denoising passes for one instruction on one image.
    Returns a dict of variance metrics and the raw stacked actions.
    """
    model = policy.model
    config = policy.config
    action_dim = config.output_features["action"].shape[0]  # 6 for SO-101

    images, img_masks, tokens, masks = prepare_inputs(
        policy, image, instruction, device
    )

    # Encode prefix once
    prefix_pad_masks, past_key_values = encode_prefix(
        model, images, img_masks, tokens, masks
    )

    chunk_size = config.chunk_size
    max_action_dim = config.max_action_dim
    actions_list = []

    for seed in range(n_seeds):
        torch.manual_seed(seed)
        if device != "cpu":
            torch.cuda.manual_seed(seed)
        noise = torch.randn(1, chunk_size, max_action_dim, device=device)
        raw_actions = denoise_with_cached_kv(
            model, prefix_pad_masks, past_key_values,
            noise, num_steps=num_steps,
        )
        actions_list.append(raw_actions[:, :, :action_dim].float().cpu())

    stacked = torch.cat(actions_list, dim=0)  # (n_seeds, chunk_size, action_dim)

    per_joint_std = stacked.std(dim=0).mean(dim=0)       # (action_dim,)
    mean_std = per_joint_std.mean().item()
    max_std = per_joint_std.max().item()

    mean_actions = stacked.mean(dim=0)                    # (chunk_size, action_dim)
    action_magnitude = mean_actions[:10].norm(dim=-1).mean().item()

    return {
        "mean_std": mean_std,
        "max_std": max_std,
        "action_magnitude": action_magnitude,
        "per_joint_std": per_joint_std.tolist(),
        "stacked_actions": stacked,
    }


# --------------------------------------------------------------------------- #
# Pretty-print results table
# --------------------------------------------------------------------------- #

def classify(mean_std, action_magnitude, std_threshold=0.10, mag_threshold=0.05):
    if mean_std < std_threshold and action_magnitude > mag_threshold:
        return "RECOGNISED"
    elif mean_std > std_threshold * 2:
        return "UNRECOGNISED"
    else:
        return "AMBIGUOUS"


def print_results(results):
    header = (f"{'Label':<25} {'Prompt':<40} "
              f"{'Mean Std':>10} {'Max Std':>10} {'Action Mag':>12} {'Verdict':<15}")
    print()
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for label, data in results.items():
        prompt = PROMPTS[label]
        verdict = classify(data["mean_std"], data["action_magnitude"])
        marker = "*" if label in TRAINING_LABELS else " "
        print(
            f"{marker}{label:<24} {prompt:<40} "
            f"{data['mean_std']:>10.6f} {data['max_std']:>10.6f} "
            f"{data['action_magnitude']:>12.6f} {verdict:<15}"
        )

    print("=" * len(header))
    print("  * = training label")
    print()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Denoising consistency test for VLA vocabulary sensitivity"
    )
    parser.add_argument("--checkpoint",
                        help="Path or HF repo id for the finetuned PI05 checkpoint", default="ant0nh/pi05_500_30k")
    parser.add_argument("--image_path", default=None,
                        help="Path to a saved scene image (PNG/JPG). "
                             "If omitted, captures from RealSense.")
    parser.add_argument("--n_seeds", type=int, default=10,
                        help="Number of noise seeds per prompt (default: 10)")
    parser.add_argument("--num_steps", type=int, default=10,
                        help="Flow-matching denoising steps (default: 10)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompts", nargs="*", default=None,
                        help="Subset of prompt labels to test (default: all)")
    args = parser.parse_args()

    log.info("Loading policy from %s ...", args.checkpoint)
    policy = load_policy(args.checkpoint, args.device)
    log.info("Policy loaded.")

    # --- Acquire scene image ---
    if args.image_path:
        log.info("Loading image from %s", args.image_path)
        image = load_scene_image(args.image_path)
    else:
        log.info("Capturing live frame from RealSense ...")
        from vlm import RealSenseCamera, capture_scene
        cam = RealSenseCamera()
        image = capture_scene(cam)
        cam.stop()
        log.info("Frame captured.")

    # --- Select prompts ---
    if args.prompts:
        selected = OrderedDict(
            (k, v) for k, v in PROMPTS.items() if k in args.prompts
        )
        if not selected:
            log.error("None of the requested labels found in PROMPTS: %s",
                      args.prompts)
            return
    else:
        selected = PROMPTS

    # --- Run consistency measurements ---
    results = OrderedDict()
    total = len(selected)

    for i, (label, prompt) in enumerate(selected.items(), 1):
        log.info("[%d/%d] Measuring: '%s'  (%s)", i, total, prompt, label)
        t0 = time.time()

        metrics = measure_consistency(
            policy, image, prompt,
            n_seeds=args.n_seeds,
            num_steps=args.num_steps,
            device=args.device,
        )

        elapsed = time.time() - t0
        log.info(
            "  mean_std=%.6f  max_std=%.6f  action_mag=%.6f  (%.1fs)",
            metrics["mean_std"], metrics["max_std"],
            metrics["action_magnitude"], elapsed,
        )
        results[label] = metrics

    print_results(results)


if __name__ == "__main__":
    main()
