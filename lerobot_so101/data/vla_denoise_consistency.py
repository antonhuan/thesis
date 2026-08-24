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

The VLA is conditioned on an initial joint pose (folded into the training
Task/State/Action prompt); it defaults to the home pose, override with
--initial_pose PAN LIFT ELBOW WFLEX WROLL GRIP, or --no_state for the old
stateless prompt.

Plots (per-joint dispersion + end-effector) are written to --plot_dir by
default; pass --no_plots to skip, and --urdf to point at so101_new_calib.urdf
for the end-effector figure (auto-detected if omitted).
"""

import argparse
import logging
import time
from collections import OrderedDict

import torch

from vla_embedding_utils import (
    build_preprocessor,
    load_policy,
    load_scene_image,
    prepare_inputs,
    prepare_inputs_stateful,
)

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

# Default initial joint pose the VLA is conditioned on (motor order:
# shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper).
# Values are the rest/home pose from robot_client_loop.py (arm in degrees,
# gripper 0-100 %); copied rather than imported to avoid pulling in the gRPC /
# hardware dependencies of that module.
HOME_POSE = [-4.571428571428571, -101.49450549450549, 91.91208791208791,
             74.28571428571429, -0.7472527472527473, 1.3013698630136987]


# --------------------------------------------------------------------------- #
# Core: cached prefix + denoising loop
# --------------------------------------------------------------------------- #

def _to_4d_additive_mask(att_2d_masks, dtype):
    """Boolean 2D [B,N,N] mask -> additive float 4D [B,1,N,N] mask.

    Mirrors PI05Pytorch._prepare_attention_masks_4d without depending on it,
    so the script runs against any lerobot version (some don't expose the
    private helper).
    """
    m = att_2d_masks[:, None, :, :]
    neg = torch.finfo(dtype).min
    return torch.where(
        m,
        torch.zeros((), dtype=dtype, device=m.device),
        torch.full((), neg, dtype=dtype, device=m.device),
    )


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
    model_dtype = next(model.paligemma_with_expert.parameters()).dtype
    prefix_att_2d_masks_4d = _to_4d_additive_mask(prefix_att_2d_masks, model_dtype)

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
                        num_steps=10, device="cuda",
                        preprocessor=None, initial_pose=None):
    """
    Run *n_seeds* denoising passes for one instruction on one image.
    Returns a dict of variance metrics and the raw stacked actions.

    When a ``preprocessor`` is given the policy is conditioned on ``initial_pose``
    via the training-faithful Task/State/Action prompt; otherwise it falls back
    to the bare, stateless prompt.
    """
    model = policy.model
    config = policy.config
    action_dim = config.output_features["action"].shape[0]  # 6 for SO-101

    if preprocessor is not None:
        images, img_masks, tokens, masks = prepare_inputs_stateful(
            policy, preprocessor, image, instruction, initial_pose, device
        )
    else:
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
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "bfloat16"],
                        help="Model precision (default: float32). 'bfloat16' "
                             "roughly halves load time / GPU memory.")
    parser.add_argument("--prompts", nargs="*", default=None,
                        help="Subset of prompt labels to test (default: all)")
    parser.add_argument("--plot_dir", default="denoise_plots",
                        help="Directory for output plots (default: denoise_plots)")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip generating plots")
    parser.add_argument("--urdf", default=None,
                        help="Path to so101_new_calib.urdf for the end-effector "
                             "plot (auto-detected if omitted)")
    parser.add_argument("--initial_pose", type=float, nargs=6, default=None,
                        metavar=("PAN", "LIFT", "ELBOW", "WFLEX", "WROLL", "GRIP"),
                        help="Initial joint pose (6 values, motor order, arm in "
                             "degrees / gripper 0-100) the VLA is conditioned on. "
                             "Defaults to the home pose.")
    parser.add_argument("--no_state", action="store_true",
                        help="Disable state conditioning (bare stateless prompt, "
                             "the old behaviour).")
    args = parser.parse_args()

    log.info("Loading policy from %s (dtype=%s) ...", args.checkpoint, args.dtype)
    policy = load_policy(args.checkpoint, args.device, args.dtype)
    log.info("Policy loaded.")

    # State conditioning: route inputs through the checkpoint's real preprocessor
    # so the VLA sees the Task/State/Action prompt it was trained on.
    preprocessor = None
    initial_pose = args.initial_pose if args.initial_pose is not None else HOME_POSE
    if args.no_state:
        log.info("State conditioning disabled (--no_state); using bare prompts.")
    else:
        preprocessor = build_preprocessor(args.checkpoint, policy.config)
        if preprocessor is not None:
            log.info("Conditioning on initial pose: %s",
                     [round(v, 2) for v in initial_pose])

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
            preprocessor=preprocessor,
            initial_pose=initial_pose,
        )

        elapsed = time.time() - t0
        log.info(
            "  mean_std=%.6f  max_std=%.6f  action_mag=%.6f  (%.1fs)",
            metrics["mean_std"], metrics["max_std"],
            metrics["action_magnitude"], elapsed,
        )
        results[label] = metrics

    print_results(results)

    # --- Plots ---
    if not args.no_plots:
        from pathlib import Path

        from vla_plots import (
            build_action_unnormalizer,
            build_kinematics,
            plot_end_effector,
            plot_joint_distributions,
        )

        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        log.info("Writing plots to %s/ ...", plot_dir)

        # Real (degree/percent) units where the checkpoint stats load; else the
        # plots stay in normalized space and the end-effector plot is skipped.
        unnorm = build_action_unnormalizer(args.checkpoint, policy.config)

        plot_joint_distributions(
            results, PROMPTS, TRAINING_LABELS,
            plot_dir / "joint_distributions.png",
            unnorm=unnorm, classify=classify,
        )

        if unnorm is not None:
            kin = build_kinematics(args.urdf)
            if kin is not None:
                plot_end_effector(
                    results, PROMPTS, TRAINING_LABELS, kin, unnorm,
                    plot_dir / "end_effector_3d.png",
                    classify=classify,
                )
        else:
            log.warning("End-effector plot skipped (need unnormalized joint "
                        "angles).")


if __name__ == "__main__":
    main()
