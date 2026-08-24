"""
Plotting for the PI0.5 denoising-consistency test.

Two visualisations are produced from the per-prompt stacked action samples
(shape (n_seeds, chunk_size, action_dim)) that ``measure_consistency`` returns:

  1. plot_joint_distributions
        One figure, six panels (one per SO-101 joint). Each panel shows a violin
        per prompt of the *per-seed deviation from the mean trajectory* — i.e. how
        much the sampled joint value scatters across noise seeds at each timestep.
        A prompt the VLA recognises produces a tight attractor -> narrow violin;
        an unrecognised prompt scatters -> wide violin.

  2. plot_end_effector
        Forward-kinematics of each sampled joint chunk into 3D gripper positions,
        drawn as one small-multiple 3D panel per prompt (seed trajectories + end
        points, shared axes). Recognised prompts land in a tight bundle.

Actions leave the policy in quantile-normalized space. Where the checkpoint's
postprocessor can be loaded we unnormalize to real units (arm joints in degrees,
gripper 0-100); the end-effector plot *requires* those real degrees and is
skipped with a warning otherwise. Forward kinematics needs ``placo`` and an
SO-101 URDF; it is likewise skipped (never fatal) if either is missing.
"""

import logging
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)

# SO-101 motor order (see pi05_client.py / so_follower.py). Arm joints are in
# degrees, gripper in 0-100 %. Forward kinematics uses only the five arm joints.
JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
N_ARM = 5

# SO-101 arm motors in order (gripper excluded from FK).
MOTOR_ARM_ORDER = JOINT_NAMES[:N_ARM]

# Published SO-101 URDFs name the revolute joints in one of two schemes and do
# NOT list them in a reliable order: the SO-ARM100 URDF names them after the
# motors ("shoulder_pan"..), while other copies use generic names and may list
# them tip->base. We therefore resolve each motor to its URDF joint *by name*
# (identity if present, else this alias), preserving motor order so the joint
# values are fed to placo against the right joints regardless of document order.
# NOTE: sign / zero-offset conventions between the Feetech motor degrees and the
# URDF zero pose are not reconciled here, so absolute EE geometry is approximate
# — the plot is for *relative* spread (tight vs dispersed), which is
# convention-independent.
URDF_JOINT_ALIASES = {
    "shoulder_pan": "Rotation",
    "shoulder_lift": "Pitch",
    "elbow_flex": "Elbow",
    "wrist_flex": "Wrist_Pitch",
    "wrist_roll": "Wrist_Roll",
}

# End-effector frame candidates, tried in order (published URDFs differ).
EE_FRAME_CANDIDATES = ["gripper_frame_link", "gripper"]


# --------------------------------------------------------------------------- #
# Unnormalization (quantile stats from the checkpoint postprocessor)
# --------------------------------------------------------------------------- #

def build_action_unnormalizer(checkpoint, config):
    """Return a callable ``actions_norm -> actions_real`` using the checkpoint's
    QUANTILES action stats, or ``None`` if they can't be loaded (caller then
    falls back to plotting in normalized space).

    Formula matches lerobot's ``UnnormalizerProcessorStep`` for QUANTILES:
        x_real = (x_norm + 1) / 2 * (q99 - q01) + q01
    """
    try:
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.processor.normalize_processor import UnnormalizerProcessorStep
        from lerobot.utils.constants import ACTION
    except Exception as e:  # noqa: BLE001 - optional/looser API across versions
        log.warning("lerobot processor API unavailable (%s); plots stay in "
                    "normalized units.", e)
        return None

    try:
        _, post = make_pre_post_processors(config, pretrained_path=str(checkpoint))
        step = next(
            (s for s in post.steps if isinstance(s, UnnormalizerProcessorStep)),
            None,
        )
        if step is None:
            raise RuntimeError("no UnnormalizerProcessorStep in postprocessor")
        stats = getattr(step, "stats", None) or {}
        astats = stats.get(ACTION) or stats.get("action")
        if not astats or "q01" not in astats or "q99" not in astats:
            raise RuntimeError("action q01/q99 stats missing (non-QUANTILES norm?)")
        q01 = torch.as_tensor(np.asarray(astats["q01"]), dtype=torch.float32).flatten()
        q99 = torch.as_tensor(np.asarray(astats["q99"]), dtype=torch.float32).flatten()
    except Exception as e:  # noqa: BLE001
        log.warning("Could not load unnormalization stats (%s); plots stay in "
                    "normalized units.", e)
        return None

    def unnorm(actions):
        a = torch.as_tensor(actions, dtype=torch.float32)
        d = min(a.shape[-1], q01.shape[-1])
        out = a.clone().float()
        out[..., :d] = (a[..., :d] + 1.0) * (q99[:d] - q01[:d]) / 2.0 + q01[:d]
        return out

    log.info("Loaded quantile unnormalization stats from checkpoint.")
    return unnorm


# --------------------------------------------------------------------------- #
# Forward kinematics (optional: needs placo + an SO-101 URDF)
# --------------------------------------------------------------------------- #

def _resolve_urdf(urdf_path):
    """Return a usable URDF path, or None. Honours an explicit path first, then
    globs common on-disk locations for ``so101_new_calib.urdf``."""
    if urdf_path:
        p = Path(urdf_path).expanduser()
        if p.is_file():
            return p
        log.warning("--urdf %s does not exist.", p)
        return None
    for root in (Path.home() / ".cache", Path.home() / ".local", Path.home()):
        try:
            hit = next(root.rglob("so101_new_calib.urdf"), None)
        except Exception:  # noqa: BLE001 - permission errors while globbing
            hit = None
        if hit is not None:
            log.info("Auto-detected SO-101 URDF: %s", hit)
            return hit
    return None


def _parse_urdf_revolute_joints(urdf_path):
    """Return the set of the URDF's revolute/continuous joint names. Order in the
    document is NOT reliable (some copies list tip->base), so callers must match
    joints by name, not position."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.parse(str(urdf_path)).getroot()
    except Exception as e:  # noqa: BLE001
        log.warning("Could not parse URDF %s (%s).", urdf_path, e)
        return set()
    return {j.get("name") for j in root.findall("joint")
            if j.get("type") in ("revolute", "continuous")}


def _resolve_arm_joint_names(urdf_path):
    """Map the five arm motors (in motor order) to this URDF's joint names, by
    name. Returns the ordered URDF joint names, or None if the URDF is missing an
    expected joint (EE plot is then skipped)."""
    urdf_joints = _parse_urdf_revolute_joints(urdf_path)
    resolved = []
    for m in MOTOR_ARM_ORDER:
        if m in urdf_joints:
            resolved.append(m)
        elif URDF_JOINT_ALIASES.get(m) in urdf_joints:
            resolved.append(URDF_JOINT_ALIASES[m])
        else:
            log.warning("URDF %s has no joint for motor '%s' (found %s); "
                        "skipping end-effector plot.",
                        Path(urdf_path).name, m, sorted(urdf_joints))
            return None
    return resolved


def build_kinematics(urdf_path=None, target_frame_name="gripper_frame_link"):
    """Build a lerobot ``RobotKinematics`` for the SO-101, or return None (with a
    helpful warning) if placo or a URDF is unavailable — never fatal.

    Drives FK with the first five *arm* revolute joints parsed from the URDF (the
    gripper is excluded so its 0-100 value is never fed to FK as degrees), and
    picks whichever end-effector frame the URDF actually defines.
    """
    try:
        from lerobot.model import RobotKinematics
    except Exception as e:  # noqa: BLE001 - placo is an optional extra
        log.warning("RobotKinematics/placo unavailable (%s); skipping "
                    "end-effector plot. Try `pip install placo`.", e)
        return None

    urdf = _resolve_urdf(urdf_path)
    if urdf is None:
        log.warning("No SO-101 URDF found; skipping end-effector plot. Pass "
                    "--urdf /path/to/so101_new_calib.urdf")
        return None

    arm_joints = _resolve_arm_joint_names(urdf)
    if arm_joints is None:
        return None
    # Try the requested tip frame first, then known alternatives; validate each
    # with a smoke-test FK call (bad frame/joint names only raise on use).
    candidates = list(dict.fromkeys([target_frame_name] + EE_FRAME_CANDIDATES))
    for frame in candidates:
        try:
            kin = RobotKinematics(str(urdf), target_frame_name=frame,
                                  joint_names=arm_joints)
            kin.forward_kinematics(np.zeros(len(arm_joints)))
            log.info("Kinematics ready (urdf=%s, tip=%s, joints=%s).",
                     Path(urdf).name, frame, arm_joints)
            return kin
        except Exception:  # noqa: BLE001 - try next frame candidate
            continue
    log.warning("Could not build kinematics from %s (no working tip frame in "
                "%s); skipping EE plot.", urdf, candidates)
    return None


def _ee_cloud(stacked_real, kin):
    """Forward-kinematics a (S, T, A) real-units chunk into (S, T, 3) EE xyz."""
    arm = np.asarray(stacked_real[..., :N_ARM], dtype=float)
    s_n, t_n = arm.shape[0], arm.shape[1]
    ee = np.empty((s_n, t_n, 3), dtype=float)
    for s in range(s_n):
        for t in range(t_n):
            ee[s, t] = kin.forward_kinematics(arm[s, t])[:3, 3]
    return ee


# --------------------------------------------------------------------------- #
# Plot 1: per-joint distributions
# --------------------------------------------------------------------------- #

def _grid(n, ncols):
    return (n + ncols - 1) // ncols


def _violin_data_per_joint(results, labels, joint_idx, unnorm):
    """For one joint, the per-prompt list of centered per-seed deviations
    (flattened over seeds x timesteps). Centering per timestep isolates the
    seed-to-seed scatter — the consistency signal — from legitimate trajectory
    motion shared by all seeds."""
    data = []
    for label in labels:
        stacked = results[label]["stacked_actions"]          # (S, T, A), norm
        if unnorm is not None:
            stacked = unnorm(stacked)
        arr = np.asarray(stacked[..., joint_idx], dtype=float)  # (S, T)
        resid = arr - arr.mean(axis=0, keepdims=True)           # center per t
        flat = resid.reshape(-1)
        if flat.std() == 0.0:  # degenerate (e.g. 1 seed) -> nudge so kde works
            flat = flat + np.random.normal(0.0, 1e-9, flat.shape)
        data.append(flat)
    return data


def plot_joint_distributions(results, prompts, training_labels, save_path,
                             unnorm=None, classify=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(results.keys())
    units = "real units" if unnorm is not None else "normalized"
    ncols = 3
    nrows = _grid(len(JOINT_NAMES), ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for j, name in enumerate(JOINT_NAMES):
        ax = axes[j]
        data = _violin_data_per_joint(results, labels, j, unnorm)
        positions = np.arange(1, len(labels) + 1)
        parts = ax.violinplot(data, positions=positions, showextrema=False,
                              widths=0.85)
        for k, body in enumerate(parts["bodies"]):
            is_train = labels[k] in training_labels
            body.set_facecolor("#2a7fff" if is_train else "#d1495b")
            body.set_alpha(0.6)
            body.set_edgecolor("black")
            body.set_linewidth(0.4)
        ax.axhline(0.0, color="gray", lw=0.7, ls="--", alpha=0.6)
        ax.set_title(name, fontsize=11)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(f"seed deviation ({units})", fontsize=8)
        ax.grid(True, axis="y", alpha=0.25)

    for extra in range(len(JOINT_NAMES), len(axes)):
        axes[extra].axis("off")

    from matplotlib.patches import Patch
    legend = [
        Patch(facecolor="#2a7fff", alpha=0.6, label="training label"),
        Patch(facecolor="#d1495b", alpha=0.6, label="synonym / out-of-domain"),
    ]
    fig.legend(handles=legend, loc="upper right", fontsize=9)
    fig.suptitle("Per-joint action dispersion across noise seeds "
                 "(narrow = recognised / strong attractor)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# --------------------------------------------------------------------------- #
# Plot 2: end-effector 3D clouds
# --------------------------------------------------------------------------- #

def plot_end_effector(results, prompts, training_labels, kin, unnorm, save_path,
                      classify=None):
    """One 3D panel per prompt: each seed's gripper trajectory (faint line) plus
    its end point (marker), shared axis limits so tightness is comparable.
    Requires a real-units unnormalizer (FK needs degrees)."""
    if unnorm is None:
        log.warning("Skipping end-effector plot: needs real (unnormalized) "
                    "joint angles, which could not be loaded.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3d proj

    labels = list(results.keys())
    clouds = {}
    for label in labels:
        real = unnorm(results[label]["stacked_actions"])
        clouds[label] = _ee_cloud(real, kin)

    allpts = np.concatenate([c.reshape(-1, 3) for c in clouds.values()], axis=0)
    mins, maxs = allpts.min(axis=0), allpts.max(axis=0)
    pad = 0.05 * np.maximum(maxs - mins, 1e-6)
    lims = list(zip(mins - pad, maxs + pad))

    ncols = 4
    nrows = _grid(len(labels), ncols)
    fig = plt.figure(figsize=(4 * ncols, 3.6 * nrows))

    for i, label in enumerate(labels):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        c = clouds[label]
        is_train = label in training_labels
        color = "#2a7fff" if is_train else "#d1495b"
        for s in range(c.shape[0]):
            ax.plot(c[s, :, 0], c[s, :, 1], c[s, :, 2],
                    color=color, lw=0.6, alpha=0.45)
        ax.scatter(c[:, -1, 0], c[:, -1, 1], c[:, -1, 2],
                   color=color, s=10, depthshade=True)
        # end-point dispersion: std of final gripper position across seeds
        ee_spread = float(np.linalg.norm(c[:, -1, :].std(axis=0)))
        verdict = ""
        if classify is not None:
            d = results[label]
            verdict = " · " + classify(d["mean_std"], d["action_magnitude"])
        ax.set_title(f"{label}{verdict}\nEE spread={ee_spread:.3f} m", fontsize=8)
        ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
        ax.tick_params(labelsize=5)
        ax.set_xlabel("x", fontsize=6); ax.set_ylabel("y", fontsize=6)
        ax.set_zlabel("z", fontsize=6)

    fig.suptitle("End-effector trajectories per prompt (FK of sampled joint "
                 "chunks) — tight bundle = recognised", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
