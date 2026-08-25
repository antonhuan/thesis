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
Task/State/Action prompt), set by the INITIAL_POSE constant near the top of this
file; pass --no_state for the old stateless prompt.

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

# Prompts grouped by the physical item they refer to. Each group is one training
# label plus its synonym / variant prompts; the out-of-domain group holds prompts
# for objects not in the training set. Plots are produced one figure per item so a
# training label sits alongside its variants (see vla_plots.py).
ITEM_GROUPS = OrderedDict([
    ("banana", ["banana", "fruit", "yellow banana"]),
    ("toy",    ["toy", "plush toy", "stuffed animal"]),
    ("pouch",  ["pouch", "purse", "bag", "drawstring pouch", "brown cylinder"]),
    ("out-of-domain", ["mug", "laptop", "shoe"]),
])

# Initial joint pose the VLA is conditioned on (motor order: shoulder_pan,
# shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper; arm in degrees,
# gripper 0-100 %). Edit this constant to change the pose. Values are the
# rest/home pose from robot_client_loop.py, copied rather than imported to avoid
# pulling in the gRPC / hardware dependencies of that module.
INITIAL_POSE = [-4.571428571428571, -101.49450549450549, 91.91208791208791,
                74.28571428571429, -0.7472527472527473, 1.3013698630136987]

# Floor on the CV denominator (mean absolute action per joint, in normalized action
# units). Prevents the coefficient of variation from blowing up when a prompt drives
# near-zero motion -- such a collapsed prompt then reads as a *high* CV (correctly
# "not a tight attractor") instead of a spurious low absolute std.
CV_EPS = 1e-3


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

    # Coefficient of variation: seed-std normalized by the per-joint action scale.
    # Scale-free measure of attractor tightness, so prompts that command
    # different-magnitude motions are directly comparable (unlike absolute std, which
    # grows with motion size). Denominator floored by CV_EPS -- see the constant.
    per_joint_scale = mean_actions.abs().mean(dim=0)      # (action_dim,)
    cv_per_joint = per_joint_std / per_joint_scale.clamp_min(CV_EPS)
    mean_cv = cv_per_joint.mean().item()
    max_cv = cv_per_joint.max().item()

    return {
        "mean_std": mean_std,
        "max_std": max_std,
        "mean_cv": mean_cv,
        "max_cv": max_cv,
        "action_magnitude": action_magnitude,
        "per_joint_std": per_joint_std.tolist(),
        "stacked_actions": stacked,
    }


# --------------------------------------------------------------------------- #
# Pretty-print results table
# --------------------------------------------------------------------------- #

def format_results_table(results, title=None):
    """Build the per-prompt stats table for one image as a string."""
    header = (f"{'Label':<25} {'Prompt':<40} "
              f"{'Mean Std':>10} {'Max Std':>10} {'Mean CV':>10} {'Max CV':>10} "
              f"{'Action Mag':>12}")
    lines = []
    if title:
        lines.append(f"# {title}")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("-" * len(header))

    for label, data in results.items():
        prompt = PROMPTS[label]
        marker = "*" if label in TRAINING_LABELS else " "
        lines.append(
            f"{marker}{label:<24} {prompt:<40} "
            f"{data['mean_std']:>10.6f} {data['max_std']:>10.6f} "
            f"{data['mean_cv']:>10.6f} {data['max_cv']:>10.6f} "
            f"{data['action_magnitude']:>12.6f}"
        )

    lines.append("=" * len(header))
    lines.append("  * = training label")
    return "\n".join(lines)


def aggregate_results(all_results):
    """Aggregate per-image metrics across images, per prompt.

    ``all_results`` maps image_name -> {label: metrics}. Returns an OrderedDict
    label -> {<metric>_mean, <metric>_sd, n_images} for mean_std / max_std /
    action_magnitude, computed as the mean and (sample) std across the images in
    which that prompt was measured.
    """
    import statistics

    # Collect the per-image value of each metric, per label (preserve prompt order).
    per_label = OrderedDict()
    for results in all_results.values():
        for label, data in results.items():
            per_label.setdefault(label, {"mean_std": [], "max_std": [],
                                         "mean_cv": [], "max_cv": [],
                                         "action_magnitude": []})
            for m in ("mean_std", "max_std", "mean_cv", "max_cv",
                      "action_magnitude"):
                per_label[label][m].append(data[m])

    agg = OrderedDict()
    for label, metrics in per_label.items():
        entry = {"n_images": len(metrics["mean_std"])}
        for m in ("mean_std", "max_std", "mean_cv", "max_cv",
                  "action_magnitude"):
            vals = metrics[m]
            entry[f"{m}_mean"] = statistics.fmean(vals) if vals else 0.0
            entry[f"{m}_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        agg[label] = entry
    return agg


def format_aggregate_table(agg, title=None):
    """Build the across-images aggregate table (mean +/- std per metric)."""
    def cell(m, s):
        return f"{m:.6f}+-{s:.6f}"

    header = (f"{'Label':<25} {'Prompt':<40} "
              f"{'Mean Std (mean+-sd)':>22} {'Max Std (mean+-sd)':>22} "
              f"{'Mean CV (mean+-sd)':>22} {'Max CV (mean+-sd)':>22} "
              f"{'Action Mag (mean+-sd)':>24}")
    lines = []
    if title:
        lines.append(f"# {title}")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("-" * len(header))

    for label, d in agg.items():
        prompt = PROMPTS[label]
        marker = "*" if label in TRAINING_LABELS else " "
        lines.append(
            f"{marker}{label:<24} {prompt:<40} "
            f"{cell(d['mean_std_mean'], d['mean_std_sd']):>22} "
            f"{cell(d['max_std_mean'], d['max_std_sd']):>22} "
            f"{cell(d['mean_cv_mean'], d['mean_cv_sd']):>22} "
            f"{cell(d['max_cv_mean'], d['max_cv_sd']):>22} "
            f"{cell(d['action_magnitude_mean'], d['action_magnitude_sd']):>24}"
        )

    lines.append("=" * len(header))
    lines.append("  * = training label")
    return "\n".join(lines)


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
                        help="Path to a single saved scene image (PNG/JPG). "
                             "If omitted (and no --image_dir), captures from "
                             "RealSense.")
    parser.add_argument("--image_dir", default=None,
                        help="Directory of scene images to run over (PNG/JPG). "
                             "Each image gets its own table + plots, plus an "
                             "aggregated table across all images.")
    parser.add_argument("--results_file", default="denoise_results.md",
                        help="File to write the per-image and aggregated stats "
                             "tables to (default: denoise_results.md)")
    parser.add_argument("--n_seeds", type=int, default=1000,
                        help="Number of noise seeds per prompt (default: 1000)")
    parser.add_argument("--num_steps", type=int, default=10,
                        help="Flow-matching denoising steps (default: 10)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "bfloat16"],
                        help="Model precision (default: float32). 'bfloat16' "
                             "roughly halves load time / GPU memory.")
    parser.add_argument("--prompts", nargs="*", default=None,
                        help="Subset of prompt labels to test (default: all)")
    parser.add_argument("--plot_dir", default="denoise_plots",
                        help="Directory for output plots (default: denoise_plots)")
    parser.add_argument("--no_plots", action="store_true",
                        help="Skip generating plots")
    parser.add_argument("--urdf", default="so101_new_calib.urdf",
                        help="Path to so101_new_calib.urdf for the end-effector "
                             "plot (auto-detected if omitted)")
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
    if args.no_state:
        log.info("State conditioning disabled (--no_state); using bare prompts.")
    else:
        preprocessor = build_preprocessor(args.checkpoint, policy.config)
        if preprocessor is not None:
            log.info("Conditioning on initial pose: %s",
                     [round(v, 2) for v in INITIAL_POSE])

    # --- Acquire scene image(s) ---
    # images: ordered list of (name, PIL.Image). A directory yields one entry per
    # image file; a single --image_path yields one entry; otherwise a live capture.
    from pathlib import Path

    images = []
    if args.image_dir:
        img_dir = Path(args.image_dir).expanduser()
        if not img_dir.is_dir():
            log.error("--image_dir %s is not a directory", img_dir)
            return
        exts = {".png", ".jpg", ".jpeg"}
        paths = sorted(p for p in img_dir.iterdir()
                       if p.suffix.lower() in exts)
        if not paths:
            log.error("No PNG/JPG images found in %s", img_dir)
            return
        for p in paths:
            log.info("Loading image from %s", p)
            images.append((p.stem, load_scene_image(str(p))))
    elif args.image_path:
        log.info("Loading image from %s", args.image_path)
        images.append((Path(args.image_path).stem,
                       load_scene_image(args.image_path)))
    else:
        log.info("Capturing live frame from RealSense ...")
        from vlm import RealSenseCamera, capture_scene
        cam = RealSenseCamera()
        images.append(("live", capture_scene(cam)))
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

    # --- Set up plotting (unnormalizer + kinematics built once, reused per image) ---
    unnorm = kin = None
    plot_dir = None
    if not args.no_plots:
        from vla_plots import (
            _safe_name,
            build_action_unnormalizer,
            build_kinematics,
            plot_action_traces,
            plot_end_effector,
            plot_joint_variance_over_time,
        )

        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        # Real (degree/percent) units where the checkpoint stats load; else the
        # plots stay in normalized space and the end-effector plot is skipped.
        unnorm = build_action_unnormalizer(args.checkpoint, policy.config)
        if unnorm is not None:
            kin = build_kinematics(args.urdf)

    # --- Run consistency measurements, per image ---
    all_results = OrderedDict()
    total = len(selected)
    n_images = len(images)

    for img_i, (image_name, image) in enumerate(images, 1):
        log.info("=== Image [%d/%d]: %s ===", img_i, n_images, image_name)
        results = OrderedDict()

        for i, (label, prompt) in enumerate(selected.items(), 1):
            log.info("[%d/%d] Measuring: '%s'  (%s)", i, total, prompt, label)
            t0 = time.time()

            metrics = measure_consistency(
                policy, image, prompt,
                n_seeds=args.n_seeds,
                num_steps=args.num_steps,
                device=args.device,
                preprocessor=preprocessor,
                initial_pose=INITIAL_POSE,
            )

            elapsed = time.time() - t0
            log.info(
                "  mean_std=%.6f  max_std=%.6f  action_mag=%.6f  (%.1fs)",
                metrics["mean_std"], metrics["max_std"],
                metrics["action_magnitude"], elapsed,
            )
            results[label] = metrics

        all_results[image_name] = results

        table = format_results_table(results, title=image_name)
        print()
        print(table)
        print()

        # --- Plots (per image, into plot_dir/<image_name>/) ---
        if not args.no_plots:
            img_plot_dir = plot_dir / _safe_name(image_name)
            log.info("Writing plots to %s/ ...", img_plot_dir)
            plot_joint_variance_over_time(
                results, ITEM_GROUPS, TRAINING_LABELS,
                img_plot_dir / "joint_variance",
                unnorm=unnorm,
            )
            plot_action_traces(
                results, ITEM_GROUPS, TRAINING_LABELS,
                img_plot_dir / "traces",
                unnorm=unnorm,
            )
            if unnorm is not None and kin is not None:
                plot_end_effector(
                    results, ITEM_GROUPS, TRAINING_LABELS, kin, unnorm,
                    img_plot_dir / "end_effector",
                )
            elif unnorm is None:
                log.warning("End-effector plot skipped (need unnormalized joint "
                            "angles).")

    # --- Aggregate across images and write the combined results file ---
    agg = aggregate_results(all_results)
    agg_table = format_aggregate_table(agg, title=f"Aggregated ({n_images} images)")
    print()
    print(agg_table)
    print()

    sections = []
    for image_name, results in all_results.items():
        sections.append(f"## {image_name}\n\n```\n"
                        f"{format_results_table(results)}\n```")
    sections.append(f"## Aggregated ({n_images} images)\n\n```\n"
                    f"{format_aggregate_table(agg)}\n```")

    results_path = Path(args.results_file)
    results_path.write_text("\n\n".join(sections) + "\n")
    log.info("Wrote results tables to %s", results_path)


if __name__ == "__main__":
    main()
