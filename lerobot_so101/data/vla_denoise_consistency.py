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
import json
import logging
import math
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from vla_embedding_utils import (
    build_preprocessor,
    load_policy,
    load_scene_image,
    prepare_inputs,
    prepare_inputs_stateful,
    prepare_inputs_stateful_multi,
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
    # Grouped by object: each training label (should be recognised) followed by
    # its synonyms / variants (may or may not be recognised).
    # banana
    ("banana",           "put the banana on the tray"),
    ("fruit",            "put the fruit on the tray"),
    ("yellow banana",    "put the yellow banana on the tray"),
    # toy
    ("toy",              "put the toy on the tray"),
    ("plush toy",        "put the plush toy on the tray"),
    ("stuffed animal",   "put the stuffed animal on the tray"),
    # pouch
    ("pouch",            "put the pouch on the tray"),
    ("brown pouch",      "put the brown pouch on the tray"),
    ("purse",            "put the purse on the tray"),
    ("bag",              "put the bag on the tray"),
    ("brown cylinder",   "put the brown cylinder on the tray"),
    ("roll", "put the roll on the tray"),
    ("box", "put the box on the tray"),
])

TRAINING_LABELS = {"banana", "toy", "pouch"}

# Prompts grouped by the physical item they refer to. Each group is one training
# label plus its synonym / variant prompts; the out-of-domain group holds prompts
# for objects not in the training set. Plots are produced one figure per item so a
# training label sits alongside its variants (see vla_plots.py).
ITEM_GROUPS = OrderedDict([
    ("banana", ["banana", "fruit", "yellow banana"]),
    ("toy",    ["toy", "plush toy", "stuffed animal"]),
    ("pouch",  ["pouch", "brown pouch", "purse", "bag", "brown cylinder", "roll", "box"]),
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

# Peak L2 displacement (from a chunk's first action, in real/unnormalized action
# units) below which a seed's planned motion counts as "no movement" -- i.e. the
# VLA did not understand the prompt. Mirrors the orchestrator's threshold
# (vlm_robot_orchestrator.py:159, no_movement_threshold); real movement produces
# peak displacements of ~150-220, non-movement ~1-3.
NO_MOVEMENT_THRESHOLD = 10.0

# Indices of the four "planning" joints whose seed variance is most sensitive
# to label mismatch (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex).
# See vla_plots.JOINT_NAMES for the full order.
PLANNING_JOINTS = [0, 1, 2, 3]


def compute_joint_auc(variance_curve, joints=None, t_start=0):
    """Area under the seed-variance curve for selected joints.

    ``variance_curve``: (T, A) array -- per-timestep, per-joint variance across
    seeds (in real units squared when unnormalized).

    ``joints``: list of joint indices to include (default: PLANNING_JOINTS).
    ``t_start``: first timestep to include (0 = full chunk).

    Returns a scalar: the sum over selected joints of the trapezoidal AUC of
    the variance curve from ``t_start`` to the end of the chunk.  Units are
    (real-unit² · timesteps), so the absolute value is only meaningful relative
    to other labels measured on the same chunk length and joint set.
    """
    if joints is None:
        joints = PLANNING_JOINTS
    curve = variance_curve[t_start:, joints]      # (T', len(joints))
    # Trapezoidal rule per joint, then sum across joints.
    return float(np.trapz(curve, axis=0).sum())



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

    ``image`` is normally a single PIL.Image; episode-replay mode instead
    passes a dict ({camera_name: PIL.Image}), which routes through
    ``prepare_inputs_stateful_multi`` to feed every camera the checkpoint's
    image_features declares (see vla_embedding_utils.py).
    """
    model = policy.model
    config = policy.config
    action_dim = config.output_features["action"].shape[0]  # 6 for SO-101

    if preprocessor is not None:
        if isinstance(image, dict):
            images, img_masks, tokens, masks = prepare_inputs_stateful_multi(
                policy, preprocessor, image, instruction, initial_pose, device
            )
        else:
            images, img_masks, tokens, masks = prepare_inputs_stateful(
                policy, preprocessor, image, instruction, initial_pose, device
            )
    else:
        if isinstance(image, dict):
            # Stateless path has no multi-camera equivalent; degrade to a
            # single image rather than failing outright.
            fallback = image.get("front") or next(iter(image.values()))
            log.warning("--no_state doesn't support multi-camera input; "
                        "using a single image only.")
            images, img_masks, tokens, masks = prepare_inputs(
                policy, fallback, instruction, device
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


def compute_motion_metrics(stacked, unnorm, kin, threshold):
    """Peak-displacement / no-movement and end-effector-consistency metrics for one
    prompt's stacked seed chunks ``stacked`` (S, T, A).

    - Peak displacement (real action units): per seed, the max over the chunk of the
      L2 distance from that seed's first action -- matches the orchestrator's
      convention (robot_client_loop.py:646).
    - ``no_move_frac``: fraction of seeds whose peak displacement is below
      ``threshold`` (the VLA planned essentially no motion for them).
    - ``ee_consistency``: whole-trajectory spread of the gripper position across
      seeds -- the mean over timesteps of the across-seed xyz-std L2 norm, in metres.

    Metrics that need the (possibly missing) unnormalizer / kinematics come back as
    NaN. Also returns the FK end-effector cloud (S, T, 3), or ``None``, so callers
    can reuse it instead of recomputing FK.
    """
    from vla_plots import _ee_cloud

    nan = float("nan")
    if unnorm is None:
        return {"ee_consistency": nan, "peak_disp_mean": nan,
                "no_move_frac": nan, "ee_cloud": None}

    real = np.asarray(unnorm(stacked), dtype=float)           # (S, T, A)
    disp = np.linalg.norm(real - real[:, :1, :], axis=-1)     # (S, T)
    peak = disp.max(axis=1)                                    # (S,)
    peak_disp_mean = float(peak.mean())
    no_move_frac = float((peak < threshold).mean())

    ee_consistency, ee_cloud = nan, None
    if kin is not None:
        ee_cloud = _ee_cloud(real, kin)                       # (S, T, 3)
        ee_consistency = float(
            np.linalg.norm(ee_cloud.std(axis=0), axis=-1).mean()
        )

    return {"ee_consistency": ee_consistency,
            "peak_disp_mean": peak_disp_mean,
            "no_move_frac": no_move_frac,
            "ee_cloud": ee_cloud}


def compute_ground_truth_metrics(stacked, unnorm, gt_real):
    """Compare a prompt's denoised seed cluster against the real recorded
    action chunk for that same forward-pass observation (episode-replay mode
    only; ``gt_real`` is the actual sequence of joint targets sent to the arm).

    ``stacked``: (S, T, A) normalized seed cluster for the ground-truth label.
    ``gt_real``: (T', 6) REAL-unit recorded action chunk. Compared over the
    overlapping prefix ``min(T, T')`` only, since an episode can end before a
    full chunk plays out.

    Returns a NaN-filled dict when ``unnorm`` is unavailable, mirroring
    ``compute_motion_metrics``'s convention so it composes with the same
    aggregation machinery.
    """
    nan = float("nan")
    if unnorm is None:
        return {"gt_dist_mean": nan, "gt_dist_median": nan,
                "gt_per_joint_mae": None}

    real = np.asarray(unnorm(stacked), dtype=float)      # (S, T, A)
    mean_real = real.mean(axis=0)                         # (T, A)
    gt_real = np.asarray(gt_real, dtype=float)             # (T', A)
    t = min(mean_real.shape[0], gt_real.shape[0])
    diff = mean_real[:t] - gt_real[:t]                     # (t, A)

    # Same mixed-units (degrees + gripper %) L2-per-timestep convention
    # compute_motion_metrics already uses for peak displacement.
    dist_per_t = np.linalg.norm(diff, axis=-1)             # (t,)
    return {
        "gt_dist_mean": float(dist_per_t.mean()),
        "gt_dist_median": float(np.median(dist_per_t)),
        "gt_per_joint_mae": np.abs(diff).mean(axis=0).tolist(),
    }


# --------------------------------------------------------------------------- #
# Pretty-print results table
# --------------------------------------------------------------------------- #

def _num(value, width, prec=6):
    """Right-align a float in ``width`` columns, or ``N/A`` when it is NaN/missing."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f"{'N/A':>{width}}"
    return f"{value:>{width}.{prec}f}"


def format_results_table(results, prompts=PROMPTS, title=None):
    """Build the per-prompt stats table for one image (or, in episode-replay
    mode, one record) as a string. ``prompts`` defaults to the module-level
    vocabulary; episode-replay mode passes its own group-scoped (and possibly
    ground-truth-patched) prompt dict instead."""
    header = (f"{'Label':<25} {'Prompt':<40} "
              f"{'Mean Std':>10} {'Max Std':>10} {'Mean CV':>10} {'Max CV':>10} "
              f"{'Action Mag':>12} {'EE Consist':>12} {'Peak Disp':>10} "
              f"{'No-Move %':>10} {'GT Dist':>10} "
              f"{'Jnt AUC':>10} {'Jnt AUC Late':>12} {'Jnt AUC All':>12}")
    lines = []
    if title:
        lines.append(f"# {title}")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("-" * len(header))

    for label, data in results.items():
        prompt = prompts[label]
        marker = "*" if label in TRAINING_LABELS else " "
        no_move = data.get("no_move_frac", float("nan"))
        no_move_pct = (no_move * 100.0 if not (isinstance(no_move, float)
                                               and math.isnan(no_move))
                       else float("nan"))
        lines.append(
            f"{marker}{label:<24} {prompt:<40} "
            f"{data['mean_std']:>10.6f} {data['max_std']:>10.6f} "
            f"{data['mean_cv']:>10.6f} {data['max_cv']:>10.6f} "
            f"{data['action_magnitude']:>12.6f} "
            f"{_num(data.get('ee_consistency'), 12)} "
            f"{_num(data.get('peak_disp_mean'), 10, 4)} "
            f"{_num(no_move_pct, 10, 1)} "
            f"{_num(data.get('gt_dist_mean'), 10, 4)} "
            f"{_num(data.get('joint_auc'), 10, 2)} "
            f"{_num(data.get('joint_auc_late'), 12, 2)} "
            f"{_num(data.get('joint_auc_all'), 12, 2)}"
        )

    lines.append("=" * len(header))
    lines.append("  * = training label")
    return "\n".join(lines)


def aggregate_results(all_results):
    """Aggregate per-image metrics across images, per prompt.

    ``all_results`` maps image_name -> {label: metrics}. Returns an OrderedDict
    label -> {<metric>_mean, <metric>_sd, n_images}, computed as the mean and
    (sample) std across the images in which that prompt was measured. NaN values
    (metrics that need an unavailable unnormalizer / kinematics) are dropped before
    aggregating, so a metric aggregates to NaN only when it was NaN everywhere.
    """
    import statistics

    metric_names = ("mean_std", "max_std", "mean_cv", "max_cv",
                    "action_magnitude", "ee_consistency", "peak_disp_mean",
                    "no_move_frac", "gt_dist_mean",
                    "joint_auc", "joint_auc_late", "joint_auc_all")

    # Collect the per-image value of each metric, per label (preserve prompt order).
    per_label = OrderedDict()
    for results in all_results.values():
        for label, data in results.items():
            per_label.setdefault(label, {m: [] for m in metric_names})
            for m in metric_names:
                per_label[label][m].append(data.get(m, float("nan")))

    agg = OrderedDict()
    for label, metrics in per_label.items():
        entry = {"n_images": len(metrics["mean_std"])}
        for m in metric_names:
            vals = [v for v in metrics[m]
                    if not (isinstance(v, float) and math.isnan(v))]
            entry[f"{m}_mean"] = statistics.fmean(vals) if vals else float("nan")
            entry[f"{m}_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        agg[label] = entry
    return agg


def format_aggregate_table(agg, prompts=PROMPTS, title=None):
    """Build the across-images (or, in episode-replay mode, across-records)
    aggregate table (mean +/- std per metric). ``prompts`` defaults to the
    module-level vocabulary; episode-replay mode passes its own group-scoped
    prompt dict instead."""
    def cell(m, s, prec=6):
        if isinstance(m, float) and math.isnan(m):
            return "N/A"
        return f"{m:.{prec}f}+-{s:.{prec}f}"

    header = (f"{'Label':<25} {'Prompt':<40} "
              f"{'Mean Std (mean+-sd)':>22} {'Max Std (mean+-sd)':>22} "
              f"{'Mean CV (mean+-sd)':>22} {'Max CV (mean+-sd)':>22} "
              f"{'Action Mag (mean+-sd)':>24} {'EE Consist (mean+-sd)':>24} "
              f"{'Peak Disp (mean+-sd)':>22} {'No-Move % (mean+-sd)':>22} "
              f"{'GT Dist (mean+-sd)':>22}")
    lines = []
    if title:
        lines.append(f"# {title}")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("-" * len(header))

    for label, d in agg.items():
        prompt = prompts[label]
        marker = "*" if label in TRAINING_LABELS else " "
        no_move_m = d["no_move_frac_mean"]
        no_move_pct_m = (no_move_m * 100.0 if not (isinstance(no_move_m, float)
                                                   and math.isnan(no_move_m))
                         else float("nan"))
        lines.append(
            f"{marker}{label:<24} {prompt:<40} "
            f"{cell(d['mean_std_mean'], d['mean_std_sd']):>22} "
            f"{cell(d['max_std_mean'], d['max_std_sd']):>22} "
            f"{cell(d['mean_cv_mean'], d['mean_cv_sd']):>22} "
            f"{cell(d['max_cv_mean'], d['max_cv_sd']):>22} "
            f"{cell(d['action_magnitude_mean'], d['action_magnitude_sd']):>24} "
            f"{cell(d['ee_consistency_mean'], d['ee_consistency_sd']):>24} "
            f"{cell(d['peak_disp_mean_mean'], d['peak_disp_mean_sd'], 4):>22} "
            f"{cell(no_move_pct_m, d['no_move_frac_sd'] * 100.0, 1):>22} "
            f"{cell(d['gt_dist_mean_mean'], d['gt_dist_mean_sd'], 4):>22} "
            f"{cell(d.get('joint_auc_mean', float('nan')), d.get('joint_auc_sd', 0.0), 2):>22} "
            f"{cell(d.get('joint_auc_late_mean', float('nan')), d.get('joint_auc_late_sd', 0.0), 2):>26} "
            f"{cell(d.get('joint_auc_all_mean', float('nan')), d.get('joint_auc_all_sd', 0.0), 2):>26}"
        )

    lines.append("=" * len(header))
    lines.append("  * = training label")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Episode replay: ground-truth every recorded input/output pair from
# robot_client_loop.py's vla_inputs/<NNN>_<object>/ episode directories.
# --------------------------------------------------------------------------- #

def iter_metadata_records(episode_dir):
    """Yield parsed dicts from episode_dir/metadata.jsonl, in file order."""
    path = Path(episode_dir) / "metadata.jsonl"
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_record_images(episode_dir, rec):
    """Load every PNG named in rec['images'] as {camera_name: PIL.Image}, or
    None (+ a logged warning) if any referenced file is missing/unreadable."""
    images = {}
    for name, fname in rec.get("images", {}).items():
        p = Path(episode_dir) / fname
        if not p.is_file():
            log.warning("Record missing image file %s (camera '%s'); skipping.",
                        p, name)
            return None
        images[name] = load_scene_image(str(p))
    return images


def action_chunk_to_array(action_chunk):
    """(list of per-timestep {'<joint>.pos': value} dicts) -> np.ndarray (T, 6),
    axis order == vla_plots.JOINT_NAMES -- the same order unnorm(stacked)
    produces, and the same order/units as INITIAL_POSE / a metadata record's
    own 'state' field."""
    from vla_plots import JOINT_NAMES
    return np.array(
        [[step[f"{j}.pos"] for j in JOINT_NAMES] for step in action_chunk],
        dtype=float,
    )


def episode_object_tag(ep_dir):
    """The object an episode directory is about, parsed from its <NNN>_<object>
    name (see robot_client_loop.py's _extract_object). Used both to resolve the
    episode's prompt group and to key the per-object aggregate, so the two can
    never disagree about which object an episode belongs to."""
    return Path(ep_dir).name.split("_", 1)[-1].lower()


def resolve_episode_group(ep_dir, task, prompts=PROMPTS, item_groups=ITEM_GROUPS,
                          label="actual"):
    """Scope the vocabulary sweep for one episode to just its own object group
    (e.g. a "put the pouch on the tray" episode only sweeps pouch variants, not
    banana/toy too), and resolve the ground-truth label within it.

    Returns (group_prompts, group_item_groups, actual_label):
      - group_prompts: an OrderedDict of just the resolved group's labels
        (patched with the literal recorded `task` if its exact wording isn't
        already one of them).
      - group_item_groups: an OrderedDict with a single entry (the resolved
        group), for the plotting functions' _present_groups filtering.
      - actual_label: the label to use as ground truth for every record in
        this episode.

    Resolution order:
      1. Parse the object tag straight from the episode folder name
         (<NNN>_<object>, see robot_client_loop.py's _extract_object) and
         match it case-insensitively against item_groups' keys.
      2. If that doesn't match any known group (a genuinely new object, or
         the folder-naming heuristic grabbed an adjective instead of a noun),
         fall back to a standalone one-member group containing just the
         literal recorded task, so the episode still gets tested.

    Never mutates the module-level PROMPTS/ITEM_GROUPS.
    """
    object_tag = episode_object_tag(ep_dir)
    group_key = next(
        (g for g in item_groups if g.lower() == object_tag), None
    )

    if group_key is not None:
        group_labels = [l for l in item_groups[group_key] if l in prompts]
        missing = [l for l in item_groups[group_key] if l not in prompts]
        if missing:
            log.warning("ITEM_GROUPS[%r] lists label(s) %s with no matching "
                        "PROMPTS entry; skipping them.", group_key, missing)
        group_prompts = OrderedDict((l, prompts[l]) for l in group_labels)
        match = next((l for l in group_labels if prompts[l] == task), None)
        if match is not None:
            return group_prompts, OrderedDict([(group_key, group_labels)]), match

        key = label
        n = 2
        while key in group_prompts:
            key = f"{label}_{n}"
            n += 1
        group_prompts[key] = task
        group_labels = group_labels + [key]
        return group_prompts, OrderedDict([(group_key, group_labels)]), key

    log.warning("Episode %s: object tag '%s' doesn't match any known "
                "ITEM_GROUPS key; testing the recorded task alone, without "
                "sibling-variant context.", Path(ep_dir).name, object_tag)
    return (OrderedDict([(label, task)]),
            OrderedDict([(label, [label])]),
            label)


def run_episode(ep_dir, policy, preprocessor, unnorm, kin, args, run_stamp):
    """Replay every metadata.jsonl record in one episode directory: sweep the
    episode's own object-group vocabulary (see resolve_episode_group) against
    that record's real image(s)/state, and compare the denoised cluster for
    the ground-truth label against the real recorded action_chunk.

    Returns None when nothing was processed, else a dict describing this
    episode (object tag, per-record results, resolved prompts/item groups) so
    the caller can roll every episode of one object up into a per-object
    aggregate -- see write_object_aggregate.
    """
    from vla_plots import (
        plot_action_traces,
        plot_end_effector,
        plot_joint_variance_over_time,
        _safe_name as _safe,
    )

    ep_dir = Path(ep_dir)
    episode_name = ep_dir.name
    log.info("=== Episode: %s ===", episode_name)

    records = list(iter_metadata_records(ep_dir))
    if not records:
        log.warning("Episode %s: no records to process.", episode_name)
        return None

    # The task (and therefore the object group) is fixed for the whole
    # episode, so this is resolved once and reused for every record.
    base_group_prompts, base_item_groups, base_actual_label = resolve_episode_group(
        ep_dir, records[0]["task"], label=args.episode_actual_label,
    )
    if args.prompts:
        base_group_prompts = OrderedDict(
            (k, v) for k, v in base_group_prompts.items()
            if k in args.prompts or k == base_actual_label
        )

    n_no_chunk = n_missing_img = n_missing_cam = 0
    all_results = OrderedDict()
    # record_id -> {label: (T, A) per-joint seed variance}. Kept alongside the
    # metrics because the caller averages these across a whole object's
    # episodes for the per-object figure (see write_object_aggregate); they are
    # ~500x smaller than the seed tensors they are derived from, so unlike
    # stacked_actions they are cheap to hold for every record of every episode.
    var_curves = OrderedDict()

    for i, rec in enumerate(records):
        if rec.get("action_chunk") is None or rec.get("action_timesteps") is None:
            n_no_chunk += 1
            continue

        images = load_record_images(ep_dir, rec)
        if images is None:
            n_missing_img += 1
            continue

        record_id = f"{rec.get('timestep', i):05d}"
        prompts, item_groups, actual_label = resolve_episode_group(
            ep_dir, rec["task"], label=args.episode_actual_label,
        )
        if args.prompts:
            prompts = OrderedDict(
                (k, v) for k, v in prompts.items()
                if k in args.prompts or k == actual_label
            )
        gt_real = action_chunk_to_array(rec["action_chunk"])

        results = OrderedDict()
        record_curves = OrderedDict()
        try:
            for label, prompt in prompts.items():
                log.info("[%s] Measuring: '%s'  (%s)", record_id, prompt, label)
                t0 = time.time()
                metrics = measure_consistency(
                    policy, images, prompt,
                    n_seeds=args.n_seeds,
                    num_steps=args.num_steps,
                    device=args.device,
                    preprocessor=preprocessor,
                    initial_pose=rec["state"],
                )
                motion = compute_motion_metrics(
                    metrics["stacked_actions"], unnorm, kin,
                    args.no_movement_threshold,
                )
                metrics.update(motion)
                # Compared against the one real recorded action_chunk for
                # every label, not just actual_label: the ground-truth
                # label should score low (model reconstructs real behavior)
                # while the other candidate prompts scoring higher is itself
                # a useful vocabulary-discrimination signal. Cheap -- pure
                # array math on the already-generated stacked_actions.
                metrics.update(compute_ground_truth_metrics(
                    metrics["stacked_actions"], unnorm, gt_real
                ))
                # Per-joint seed variance over the chunk, in the same units the
                # plot would use, so the per-object aggregate can average these
                # instead of re-pooling raw seed chunks across scenes.
                st = metrics["stacked_actions"]
                real = unnorm(st) if unnorm is not None else st
                record_curves[label] = np.asarray(real, dtype=float).var(axis=0)
                metrics["joint_auc"] = compute_joint_auc(record_curves[label])
                metrics["joint_auc_late"] = compute_joint_auc(
                    record_curves[label], t_start=record_curves[label].shape[0] // 2)
                metrics["joint_auc_all"] = compute_joint_auc(
                    record_curves[label], joints=list(range(record_curves[label].shape[1])))

                elapsed = time.time() - t0
                log.info("  mean_std=%.6f  action_mag=%.6f  (%.1fs)",
                         metrics["mean_std"], metrics["action_magnitude"], elapsed)
                results[label] = metrics
        except ValueError as e:
            log.warning("Record %s: %s; skipping (missing camera).", record_id, e)
            n_missing_cam += 1
            continue

        all_results[record_id] = results
        var_curves[record_id] = record_curves

        table = format_results_table(results, prompts=prompts, title=record_id)
        print()
        print(table)
        print()

        rec_dir = Path(args.plot_dir) / run_stamp / episode_name / record_id
        rec_dir.mkdir(parents=True, exist_ok=True)
        if not args.no_plots:
            (rec_dir / args.results_file).write_text(
                f"## {record_id}\n\n```\n{table}\n```\n"
            )

        chunk_dir = rec_dir / "action_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for label, data in results.items():
            np.save(chunk_dir / f"{_safe(label)}.npy",
                    data["stacked_actions"].cpu().numpy())

        gt_dir = rec_dir / "ground_truth"
        gt_dir.mkdir(parents=True, exist_ok=True)
        np.save(gt_dir / f"{_safe(actual_label)}_gt_chunk.npy", gt_real)
        (gt_dir / f"{_safe(actual_label)}_gt_metrics.json").write_text(json.dumps({
            k: results[actual_label].get(k) for k in
            ("gt_dist_mean", "gt_dist_median", "gt_per_joint_mae")
        }, indent=2))

        if not args.no_plots:
            gt_overlay = {actual_label: gt_real} if unnorm is not None else None
            plot_joint_variance_over_time(
                results, item_groups, TRAINING_LABELS,
                rec_dir / "joint_variance", unnorm=unnorm,
            )
            plot_action_traces(
                results, item_groups, TRAINING_LABELS,
                rec_dir / "traces", unnorm=unnorm, ground_truth=gt_overlay,
            )
            if unnorm is not None and kin is not None:
                clouds = {label: data.get("ee_cloud")
                          for label, data in results.items()}
                plot_end_effector(
                    results, item_groups, TRAINING_LABELS, kin, unnorm,
                    rec_dir / "end_effector", clouds=clouds,
                    ground_truth=gt_overlay,
                )

        for data in results.values():
            data.pop("ee_cloud", None)

    log.info(
        "Episode %s: %d processed, %d skipped (no action_chunk), "
        "%d skipped (missing image), %d skipped (missing camera)",
        episode_name, len(all_results), n_no_chunk, n_missing_img, n_missing_cam,
    )

    if not all_results:
        return None

    # --- Episode-level aggregate. Pooling the ground-truth label's seed
    # samples is only valid *within* this one episode (same recorded task
    # throughout) -- never pooled across episodes by any caller of this
    # function, see main()'s --episode_root loop. ---
    agg = aggregate_results(all_results)
    agg_dir = Path(args.plot_dir) / run_stamp / episode_name / "aggregated"
    agg_dir.mkdir(parents=True, exist_ok=True)
    agg_table = format_aggregate_table(
        agg, prompts=base_group_prompts,
        title=f"{episode_name} aggregated ({len(all_results)} records)",
    )
    print()
    print(agg_table)
    print()
    if not args.no_plots:
        (agg_dir / args.results_file).write_text(f"```\n{agg_table}\n```\n")

        pooled = OrderedDict()
        for results in all_results.values():
            for label, data in results.items():
                pooled.setdefault(label, []).append(data["stacked_actions"])
        agg_results = OrderedDict(
            (label, {"stacked_actions": torch.cat(chunks, dim=0)})
            for label, chunks in pooled.items()
        )
        plot_joint_variance_over_time(
            agg_results, base_item_groups, TRAINING_LABELS,
            agg_dir / "joint_variance", unnorm=unnorm,
        )
        plot_action_traces(
            agg_results, base_item_groups, TRAINING_LABELS,
            agg_dir / "traces", unnorm=unnorm,
        )
        if unnorm is not None and kin is not None:
            plot_end_effector(
                agg_results, base_item_groups, TRAINING_LABELS, kin, unnorm,
                agg_dir / "end_effector",
            )

    # The seed tensors have served every plot that needs them; drop them before
    # handing results back, so the caller can hold one object's worth of
    # episodes without carrying hundreds of MB of chunks it will never use.
    for res in all_results.values():
        for metrics in res.values():
            metrics.pop("stacked_actions", None)

    return {
        "episode": episode_name,
        "object": episode_object_tag(ep_dir),
        "results": all_results,          # record_id -> {label: scalar metrics}
        "variance": var_curves,          # record_id -> {label: (T, A) array}
        "prompts": base_group_prompts,
        "item_groups": base_item_groups,
    }


def write_object_aggregate(object_tag, entry, unnorm, kin, args, run_stamp):
    """Roll every record of every episode for one object into a single table
    plus one joint-variance figure (e.g. all five banana episodes together).

    Valid to pool because one object's episodes all share the same recorded
    task, and so resolve to the same prompt labels -- unlike pooling across
    objects, where one label would span unrelated trajectories. Traces and
    end-effector clouds are deliberately not aggregated here: those show where
    the arm actually went, which is a different scene in every episode.
    """
    from vla_plots import plot_joint_variance_over_time, _safe_name

    results_by_record = entry["results"]
    if not results_by_record:
        return

    agg = aggregate_results(results_by_record)
    out_dir = Path(args.plot_dir) / run_stamp / "by_object" / _safe_name(object_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg_table = format_aggregate_table(
        agg, prompts=entry["prompts"],
        title=f"{object_tag} — all episodes "
              f"({entry['n_episodes']} episodes, {len(results_by_record)} records)",
    )
    print()
    print(agg_table)
    print()
    (out_dir / args.results_file).write_text(f"```\n{agg_table}\n```\n")
    log.info("Wrote per-object aggregate for '%s' to %s/", object_tag, out_dir)

    if args.no_plots or not entry["variance"]:
        return

    # Mean of the per-record variance curves: "at timestep t of a chunk, how
    # much do seeds typically disagree for this object".
    mean_curves = OrderedDict(
        (label, np.mean(np.stack(curves, axis=0), axis=0))
        for label, curves in entry["variance"].items()
    )
    # Save raw mean-variance curves so downstream analysis can compute
    # metrics (AUC, slopes, etc.) without re-running denoising.
    curves_dir = out_dir / "variance_curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    for label, curve in mean_curves.items():
        np.save(curves_dir / f"{_safe_name(label)}.npy", curve)
    log.info("Saved per-joint variance curves to %s/", curves_dir)

    plot_joint_variance_over_time(
        {label: {} for label in mean_curves},
        entry["item_groups"], TRAINING_LABELS,
        out_dir / "joint_variance", unnorm=unnorm,
        variance_curves=mean_curves,
    )


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
    parser.add_argument("--no_movement_threshold", type=float,
                        default=NO_MOVEMENT_THRESHOLD,
                        help="Peak L2 displacement (real action units) below which "
                             "a seed's chunk counts as no movement "
                             f"(default: {NO_MOVEMENT_THRESHOLD}, mirrors the "
                             "orchestrator).")
    parser.add_argument("--episode_dir", default=None,
                        help="Path to one robot_client_loop.py episode directory "
                             "(vla_inputs/<NNN>_<object>/, containing "
                             "metadata.jsonl + PNGs). Replays every must_go "
                             "record: sweeps the episode's own object-group "
                             "vocabulary (see ITEM_GROUPS), with the record's "
                             "own recorded task guaranteed to be included as a "
                             "ground-truth label, and compares the denoised "
                             "cluster against the real recorded action_chunk. "
                             "Mutually exclusive with --episode_root, "
                             "--image_dir, --image_path, and live capture. "
                             "--n_seeds's default (1000) is very expensive "
                             "per-record; lower it for replay runs.")
    parser.add_argument("--episode_root", default=None,
                        help="Path to a vla_inputs/ directory; replays every "
                             "subdirectory containing a metadata.jsonl as its "
                             "own episode (see --episode_dir). Mutually "
                             "exclusive with --episode_dir, --image_dir, "
                             "--image_path.")
    parser.add_argument("--episode_actual_label", default="actual",
                        help="Label key to use when the recorded task doesn't "
                             "match any existing PROMPTS entry within its "
                             "resolved object group, or when the episode's "
                             "object tag doesn't match any known group at all "
                             "(default: 'actual'; auto-suffixed on collision).")
    args = parser.parse_args()

    log.info("Loading policy from %s (dtype=%s) ...", args.checkpoint, args.dtype)
    policy = load_policy(args.checkpoint, args.device, args.dtype)
    log.info("Policy loaded.")
    log.info("Checkpoint image_features: %s", list(policy.config.image_features))

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

    # Stamp each run with the local date/time so reruns (plots and the results
    # file) land in their own path instead of overwriting the previous run.
    run_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")

    # --- Unnormalizer + kinematics (built once, reused per image/episode).
    # Needed for the peak-displacement / no-movement and end-effector-consistency
    # metrics as well as the plots, so they are built regardless of --no_plots.
    # Real (degree/percent) units where the checkpoint stats load; otherwise
    # unnorm is None and those metrics / the EE plot fall back to N/A /
    # normalized space. Moved up here (was built after image acquisition) so
    # episode-replay mode below can use them too. ---
    from vla_plots import build_action_unnormalizer, build_kinematics
    unnorm = build_action_unnormalizer(args.checkpoint, policy.config)
    kin = build_kinematics(args.urdf) if unnorm is not None else None

    # --- Episode-replay mode: run every metadata.jsonl record from one or more
    # robot_client_loop.py episode directories, ground-truthed against the real
    # action_chunk each record recorded (see run_episode). ---
    if args.episode_dir or args.episode_root:
        # Recursive (rglob) rather than one level deep: robot_client_loop.py
        # nests episodes under an <object>/ subdirectory
        # (vla_inputs/<object>/<NNN>_<object>/), so --episode_root can point
        # at the whole vla_inputs/ root (sweeps every object) or at one
        # object's subdirectory (sweeps just that object) and either way
        # finds every episode regardless of nesting depth. Also picks up any
        # leftover flat-layout episode dirs from before that nesting existed.
        episode_dirs = (
            [Path(args.episode_dir)] if args.episode_dir
            else sorted(p.parent for p in Path(args.episode_root).rglob("metadata.jsonl"))
        )
        if not episode_dirs:
            log.error("No episode directories with metadata.jsonl found.")
            return

        # Accumulate every episode's records under its object, so all banana
        # episodes report together, all pouch together, etc (see
        # write_object_aggregate).
        per_object = OrderedDict()
        for ep_dir in episode_dirs:
            out = run_episode(ep_dir, policy, preprocessor, unnorm, kin, args,
                              run_stamp)
            if out is None:
                continue
            entry = per_object.setdefault(out["object"], {
                "results": OrderedDict(), "variance": OrderedDict(),
                "prompts": OrderedDict(), "item_groups": OrderedDict(),
                "n_episodes": 0,
            })
            entry["n_episodes"] += 1
            # Merge rather than overwrite: an episode whose exact task wording
            # wasn't already a label patched in its own, and dropping the
            # others would lose labels the sibling episodes did test.
            entry["prompts"].update(out["prompts"])
            for group, labels in out["item_groups"].items():
                known = entry["item_groups"].setdefault(group, [])
                known.extend(l for l in labels if l not in known)
            # Namespace by episode -- record ids restart at 00000 in every
            # episode and would otherwise silently overwrite each other.
            for record_id, res in out["results"].items():
                entry["results"][f"{out['episode']}/{record_id}"] = res
            for curves in out["variance"].values():
                for label, curve in curves.items():
                    entry["variance"].setdefault(label, []).append(curve)

        for object_tag, entry in per_object.items():
            write_object_aggregate(object_tag, entry, unnorm, kin, args,
                                   run_stamp)
        return

    # --- Acquire scene image(s) ---
    # images: ordered list of (name, PIL.Image). A directory yields one entry per
    # image file; a single --image_path yields one entry; otherwise a live capture.
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

    plot_dir = None
    if not args.no_plots:
        from vla_plots import (
            plot_action_traces,
            plot_end_effector,
            plot_joint_variance_over_time,
        )

        plot_dir = Path(args.plot_dir) / run_stamp
        plot_dir.mkdir(parents=True, exist_ok=True)
        log.info("Plots for this run go to %s/", plot_dir)

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

            motion = compute_motion_metrics(
                metrics["stacked_actions"], unnorm, kin,
                args.no_movement_threshold,
            )
            metrics.update(motion)

            elapsed = time.time() - t0
            log.info(
                "  mean_std=%.6f  max_std=%.6f  action_mag=%.6f  "
                "ee_consist=%.6f  peak_disp=%.4f  no_move=%.0f%%  (%.1fs)",
                metrics["mean_std"], metrics["max_std"],
                metrics["action_magnitude"], metrics["ee_consistency"],
                metrics["peak_disp_mean"], 100.0 * metrics["no_move_frac"],
                elapsed,
            )
            results[label] = metrics

        all_results[image_name] = results

        table = format_results_table(results, title=image_name)
        print()
        print(table)
        print()

        # This image's output directory (holds its table, plots, and the raw
        # per-seed action chunks). Built from --plot_dir even with --no_plots so
        # the chunks are always saved somewhere predictable.
        from vla_plots import _safe_name as _safe
        img_plot_dir = (
            plot_dir if plot_dir is not None
            else Path(args.plot_dir) / run_stamp
        ) / _safe(image_name)
        img_plot_dir.mkdir(parents=True, exist_ok=True)

        # --- Save every seed's raw action chunk (n_seeds x chunk x action_dim)
        # per prompt, one .npy file each. ---
        chunk_dir = img_plot_dir / "action_chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for label, data in results.items():
            np.save(chunk_dir / f"{_safe(label)}.npy",
                    data["stacked_actions"].cpu().numpy())
        log.info("Saved per-seed action chunks to %s/", chunk_dir)

        # --- Plots + table (per image, into plot_dir/<image_name>/) ---
        if not args.no_plots:
            table_path = img_plot_dir / args.results_file
            table_path.write_text(f"## {image_name}\n\n```\n{table}\n```\n")
            log.info("Wrote results table to %s", table_path)
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
                # Reuse the FK clouds already computed for the EE-consistency
                # metric so plotting doesn't re-run forward kinematics.
                clouds = {label: data.get("ee_cloud")
                          for label, data in results.items()}
                plot_end_effector(
                    results, ITEM_GROUPS, TRAINING_LABELS, kin, unnorm,
                    img_plot_dir / "end_effector", clouds=clouds,
                )
            elif unnorm is None:
                log.warning("End-effector plot skipped (need unnormalized joint "
                            "angles).")

        # Drop the cached FK clouds now that this image's EE plot is done, so they
        # don't accumulate across images (the metric values are already stored).
        for data in results.values():
            data.pop("ee_cloud", None)

    # --- Aggregate across images and write the combined results file ---
    agg = aggregate_results(all_results)
    agg_table = format_aggregate_table(agg, title=f"Aggregated ({n_images} images)")
    print()
    print(agg_table)
    print()

    # --- Aggregated plots (into plot_dir/aggregated/) ---
    # Pool every image's seed samples per prompt so the figures reflect
    # consistency across the whole test rather than a single scene. Redundant
    # with the per-image plots when there is only one image, so skip then.
    if not args.no_plots and n_images > 1:
        pooled = OrderedDict()
        for results in all_results.values():
            for label, data in results.items():
                pooled.setdefault(label, []).append(data["stacked_actions"])
        agg_results = OrderedDict(
            (label, {"stacked_actions": torch.cat(chunks, dim=0)})
            for label, chunks in pooled.items()
        )
        agg_plot_dir = plot_dir / "aggregated"
        log.info("Writing aggregated plots to %s/ ...", agg_plot_dir)
        plot_joint_variance_over_time(
            agg_results, ITEM_GROUPS, TRAINING_LABELS,
            agg_plot_dir / "joint_variance", unnorm=unnorm,
        )
        plot_action_traces(
            agg_results, ITEM_GROUPS, TRAINING_LABELS,
            agg_plot_dir / "traces", unnorm=unnorm,
        )
        if unnorm is not None and kin is not None:
            plot_end_effector(
                agg_results, ITEM_GROUPS, TRAINING_LABELS, kin, unnorm,
                agg_plot_dir / "end_effector",
            )

    sections = []
    # Per-image tables live alongside that image's plots (see loop above); only
    # fold them into the combined file when plots are disabled and they have no
    # other home.
    if args.no_plots:
        for image_name, results in all_results.items():
            sections.append(f"## {image_name}\n\n```\n"
                            f"{format_results_table(results)}\n```")
    sections.append(f"## Aggregated ({n_images} images)\n\n```\n"
                    f"{format_aggregate_table(agg)}\n```")

    # The aggregated table sits with the aggregated plots when those exist;
    # otherwise it lives at the top of this run's directory (plot_dir already
    # carries the run stamp). Without plots there is no run directory, so stamp
    # the filename instead (e.g. denoise_results_2026-08-25_14-32-05.md).
    results_arg = Path(args.results_file)
    if not args.no_plots and n_images > 1:
        results_path = plot_dir / "aggregated" / results_arg.name
    elif plot_dir is not None:
        results_path = plot_dir / results_arg.name
    else:
        results_path = results_arg.with_name(
            f"{results_arg.stem}_{run_stamp}{results_arg.suffix}"
        )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text("\n\n".join(sections) + "\n")
    log.info("Wrote results tables to %s", results_path)


if __name__ == "__main__":
    main()
