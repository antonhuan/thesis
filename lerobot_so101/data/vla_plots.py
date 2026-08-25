"""
Plotting for the PI0.5 denoising-consistency test.

Visualisations are produced from the per-prompt stacked action samples
(shape (n_seeds, chunk_size, action_dim)) that ``measure_consistency`` returns.
Prompts are grouped by the physical *item* they refer to (a training label plus
its synonym / variant prompts; see ``ITEM_GROUPS`` in the driver), and every plot
emits **one figure per item** so a training label sits alongside its variants:

  1. plot_joint_variance_over_time
        One figure per item, six panels (one per SO-101 joint). Each panel plots
        one line per prompt in the item group: the variance across noise seeds at
        each timestep of the action chunk. A prompt the VLA recognises produces a
        tight attractor -> low/flat line; an unrecognised prompt scatters -> rising.

  2. plot_action_traces
        One figure per item, six panels. For each prompt in the group, every seed's
        action trajectory (faint) plus the across-seed mean (bold), coloured per
        prompt so variants overlay for direct comparison.

  3. plot_end_effector
        Forward-kinematics of each sampled joint chunk into 3D gripper positions,
        drawn as one 3D figure per item with every prompt in the group overlaid on a
        single axis (seed trajectories + end points, shared axes across items).
        Recognised prompts land in a tight bundle.

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


def _urdf_without_meshes(urdf_path):
    """Return a path to a copy of ``urdf_path`` with every ``<mesh>`` replaced by
    an inert box primitive, so placo/pinocchio loads no external STL files.

    Forward kinematics needs only the kinematic tree (link/joint origins + axes);
    visual/collision geometry is irrelevant. This lets FK run from a bare URDF
    whose mesh assets weren't copied alongside it. Falls back to the original path
    if the rewrite fails."""
    import re
    import tempfile

    try:
        text = Path(urdf_path).read_text()
        # <mesh .../> and <mesh ...>...</mesh> -> tiny box (geometry is ignored by FK)
        text = re.sub(r"<mesh\b[^>]*/>", '<box size="0.001 0.001 0.001"/>', text)
        text = re.sub(r"<mesh\b[^>]*>.*?</mesh>",
                      '<box size="0.001 0.001 0.001"/>', text, flags=re.DOTALL)
        tmp = tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False)
        tmp.write(text)
        tmp.close()
        return tmp.name
    except Exception as e:  # noqa: BLE001 - fall back to the original URDF
        log.warning("Could not strip meshes from %s (%s); using it as-is.",
                    urdf_path, e)
        return str(urdf_path)


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

    # placo is an optional lerobot extra; RobotKinematics only raises about it
    # deep inside __init__, so check up front for a clear, actionable message.
    try:
        from lerobot.utils.import_utils import _placo_available
    except Exception:  # noqa: BLE001 - older lerobot without the flag
        _placo_available = True
    if not _placo_available:
        log.warning("placo not installed; skipping end-effector plot. "
                    "Install it with `pip install placo`.")
        return None

    urdf = _resolve_urdf(urdf_path)
    if urdf is None:
        log.warning("No SO-101 URDF found; skipping end-effector plot. Pass "
                    "--urdf /path/to/so101_new_calib.urdf")
        return None

    arm_joints = _resolve_arm_joint_names(urdf)
    if arm_joints is None:
        return None
    # Strip mesh geometry so placo needn't resolve external STL assets (FK only
    # needs the kinematic tree, which is left untouched).
    fk_urdf = _urdf_without_meshes(urdf)
    # Try the requested tip frame first, then known alternatives; validate each
    # with a smoke-test FK call (bad frame/joint names only raise on use).
    candidates = list(dict.fromkeys([target_frame_name] + EE_FRAME_CANDIDATES))
    last_err = None
    for frame in candidates:
        try:
            kin = RobotKinematics(fk_urdf, target_frame_name=frame,
                                  joint_names=arm_joints)
            kin.forward_kinematics(np.zeros(len(arm_joints)))
            log.info("Kinematics ready (urdf=%s, tip=%s, joints=%s).",
                     Path(urdf).name, frame, arm_joints)
            return kin
        except Exception as e:  # noqa: BLE001 - try next frame candidate
            last_err = e
            continue
    log.warning("Could not build kinematics from %s (tried tips %s; last error: "
                "%s); skipping EE plot.", urdf, candidates, last_err)
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
# Plot 1: per-joint variance across seeds, over the action-chunk timesteps
# --------------------------------------------------------------------------- #

def _grid(n, ncols):
    return (n + ncols - 1) // ncols


def _safe_name(label):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in label)


def _present_groups(item_groups, results):
    """Filter each item group to the prompt labels actually present in ``results``
    (a ``--prompts`` subset may drop some), preserving order and skipping empty
    groups. Returns an OrderedDict(item -> [labels])."""
    from collections import OrderedDict
    out = OrderedDict()
    for item, labels in item_groups.items():
        present = [l for l in labels if l in results]
        if present:
            out[item] = present
    return out


# --------------------------------------------------------------------------- #
# House style (publication-quality matplotlib defaults + palette)
# --------------------------------------------------------------------------- #

# Deep blue reserved for the training label so it reads as the reference in every
# figure; variants are drawn from a muted Tableau-style categorical palette that
# deliberately excludes blue so nothing competes with the training line.
_TRAINING_COLOR = "#2f5c9e"
_PALETTE = [
    "#e15759",  # red
    "#59a14f",  # green
    "#f28e2b",  # orange
    "#76b7b2",  # teal
    "#b07aa1",  # purple
    "#edc948",  # gold
    "#9c755f",  # brown
    "#ff9da7",  # pink
]

_INK = "#222222"      # primary text
_MUTED = "#6b6b6b"    # secondary text / ticks
_STYLED = False


def _set_style():
    """Apply the house matplotlib style once per process (idempotent)."""
    global _STYLED
    if _STYLED:
        return
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white",
        "figure.dpi": 120,
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                            "DejaVu Sans"],
        "font.size": 11,
        "text.color": _INK,
        "axes.facecolor": "white",
        "axes.edgecolor": "#c8c8c8",
        "axes.linewidth": 0.8,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlecolor": _INK,
        "axes.titlepad": 8,
        "axes.labelsize": 10,
        "axes.labelcolor": _MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e8e8e8",
        "grid.linewidth": 0.8,
        "xtick.color": _MUTED,
        "ytick.color": _MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "figure.titlesize": 15,
        "figure.titleweight": "bold",
    })
    _STYLED = True


def _group_colors(labels, training_labels):
    """Assign a distinct colour to each prompt in a group: the training label gets
    the reserved reference blue, variants are drawn from the categorical palette."""
    colors, i = {}, 0
    for label in labels:
        if label in training_labels:
            colors[label] = _TRAINING_COLOR
        else:
            colors[label] = _PALETTE[i % len(_PALETTE)]
            i += 1
    return colors


def _prompt_legend(fig, labels, colors, training_labels, extra=None):
    """Draw a single horizontal figure legend along the bottom.

    ``extra`` optionally maps label -> suffix string (e.g. an end-point spread) to
    append after the prompt name."""
    from matplotlib.lines import Line2D
    handles = []
    for label in labels:
        is_train = label in training_labels
        text = label + ("  (training)" if is_train else "")
        if extra and label in extra:
            text += extra[label]
        handles.append(Line2D([0], [0], color=colors[label],
                              lw=2.6 if is_train else 1.7, label=text))
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), 4), bbox_to_anchor=(0.5, 0.0),
               handlelength=1.7, columnspacing=1.8, borderaxespad=0.0)


def plot_joint_variance_over_time(results, item_groups, training_labels, save_dir,
                                  unnorm=None):
    """Per joint, the variance across noise seeds at each timestep of the action
    chunk. Keeping the time axis (rather than collapsing it) shows *where* in the
    chunk seeds agree or diverge — and, unlike a single pooled number, does not
    let temporal profile differences hide. One figure per item; within it, one
    line per prompt in the group, coloured per prompt. Recognised prompts (incl.
    the training label) stay low/flat, unrecognised ones rise."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _set_style()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    units = "real units" if unnorm is not None else "normalized"
    ncols = 3
    nrows = _grid(len(JOINT_NAMES), ncols)

    for item, labels in _present_groups(item_groups, results).items():
        colors = _group_colors(labels, training_labels)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5.2 * ncols, 3.4 * nrows))
        axes = np.atleast_1d(axes).flatten()

        for j, name in enumerate(JOINT_NAMES):
            ax = axes[j]
            for label in labels:
                stacked = results[label]["stacked_actions"]      # (S, T, A)
                if unnorm is not None:
                    stacked = unnorm(stacked)
                arr = np.asarray(stacked[..., j], dtype=float)    # (S, T)
                var_t = arr.var(axis=0)                           # (T,)
                is_train = label in training_labels
                ax.plot(np.arange(var_t.shape[0]), var_t,
                        color=colors[label],
                        lw=2.2 if is_train else 1.3,
                        alpha=1.0 if is_train else 0.85,
                        zorder=3 if is_train else 2,
                        solid_capstyle="round")
            ax.set_title(name)
            ax.set_xlabel("timestep in chunk")
            ax.margins(x=0.02)
            ax.set_ylim(bottom=0)

        # Shared y-label on the left column only, to reduce clutter.
        for row in range(nrows):
            axes[row * ncols].set_ylabel(f"seed variance ({units})")

        for extra in range(len(JOINT_NAMES), len(axes)):
            axes[extra].axis("off")

        fig.suptitle(f"Per-joint seed variance across the action chunk — {item}")
        _prompt_legend(fig, labels, colors, training_labels)
        fig.tight_layout(rect=[0, 0.05, 1, 0.96])
        out = save_dir / f"joint_variance_{_safe_name(item)}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  Saved: {out}")


# --------------------------------------------------------------------------- #
# Plot 1b: action-chunk traces (per prompt)
# --------------------------------------------------------------------------- #

def plot_action_traces(results, item_groups, training_labels, save_dir,
                       unnorm=None):
    """One figure per item: for each joint, every seed's action trajectory over the
    chunk (faint) plus the across-seed mean (bold), for each prompt in the item
    group, coloured per prompt so variants overlay. This is the actual motion the
    VLA plans — a tight bundle means a strong attractor, a frayed one means the
    prompt isn't pinning the trajectory down."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _set_style()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    units = "real units" if unnorm is not None else "normalized"
    ncols = 3
    nrows = _grid(len(JOINT_NAMES), ncols)

    for item, labels in _present_groups(item_groups, results).items():
        colors = _group_colors(labels, training_labels)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(5.2 * ncols, 3.4 * nrows))
        axes = np.atleast_1d(axes).flatten()

        for j, name in enumerate(JOINT_NAMES):
            ax = axes[j]
            for label in labels:
                stacked = results[label]["stacked_actions"]      # (S, T, A)
                if unnorm is not None:
                    stacked = unnorm(stacked)
                arr = np.asarray(stacked, dtype=float)
                s_n, t_n, _ = arr.shape
                steps = np.arange(t_n)
                is_train = label in training_labels
                # Faint per-seed spread behind the bold across-seed mean.
                for s in range(s_n):
                    ax.plot(steps, arr[s, :, j], color=colors[label],
                            lw=0.5, alpha=0.10, zorder=1)
                ax.plot(steps, arr[:, :, j].mean(axis=0),
                        color=colors[label],
                        lw=2.4 if is_train else 1.6,
                        zorder=4 if is_train else 3,
                        solid_capstyle="round")
            ax.set_title(name)
            ax.set_xlabel("timestep in chunk")
            ax.margins(x=0.02)
        for extra in range(len(JOINT_NAMES), len(axes)):
            axes[extra].axis("off")

        for row in range(nrows):
            axes[row * ncols].set_ylabel(f"action ({units})")

        fig.suptitle(f"Action-chunk trajectories — {item}")
        _prompt_legend(fig, labels, colors, training_labels)
        fig.tight_layout(rect=[0, 0.05, 1, 0.96])
        out = save_dir / f"traces_{_safe_name(item)}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  Saved: {out}")


# --------------------------------------------------------------------------- #
# Plot 2: end-effector 3D clouds
# --------------------------------------------------------------------------- #

def plot_end_effector(results, item_groups, training_labels, kin, unnorm, save_dir,
                      clouds=None):
    """One 3D figure per item: every prompt in the group overlaid on a single axis,
    each seed's gripper trajectory (faint line) plus its end point (marker),
    coloured per prompt. Axis limits are shared across all items so tightness is
    comparable between figures. Requires a real-units unnormalizer (FK needs
    degrees).

    ``clouds`` optionally supplies precomputed FK end-effector clouds
    (label -> (S, T, 3) array); any label missing from it is FK'd here. Passing the
    clouds the caller already computed (e.g. for the EE-consistency metric) avoids a
    second, expensive forward-kinematics pass."""
    if unnorm is None:
        log.warning("Skipping end-effector plot: needs real (unnormalized) "
                    "joint angles, which could not be loaded.")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3d proj

    _set_style()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    groups = _present_groups(item_groups, results)

    # FK every present prompt once (reusing any caller-supplied clouds), then
    # derive shared (global) axis limits so bundle tightness is comparable across
    # the per-item figures.
    clouds = dict(clouds) if clouds else {}
    for labels in groups.values():
        for label in labels:
            if clouds.get(label) is None:
                real = unnorm(results[label]["stacked_actions"])
                clouds[label] = _ee_cloud(real, kin)

    allpts = np.concatenate([c.reshape(-1, 3) for c in clouds.values()], axis=0)
    mins, maxs = allpts.min(axis=0), allpts.max(axis=0)
    pad = 0.05 * np.maximum(maxs - mins, 1e-6)
    lims = list(zip(mins - pad, maxs + pad))

    for item, labels in groups.items():
        colors = _group_colors(labels, training_labels)
        fig = plt.figure(figsize=(8, 7.2))
        ax = fig.add_subplot(111, projection="3d")

        spreads = {}
        for label in labels:
            c = clouds[label]
            color = colors[label]
            is_train = label in training_labels
            for s in range(c.shape[0]):
                ax.plot(c[s, :, 0], c[s, :, 1], c[s, :, 2],
                        color=color, lw=0.6, alpha=0.30 if is_train else 0.22)
            ax.scatter(c[:, -1, 0], c[:, -1, 1], c[:, -1, 2],
                       color=color, s=18 if is_train else 12,
                       edgecolors="white", linewidths=0.3, depthshade=True)
            # end-point dispersion: std of final gripper position across seeds
            spreads[label] = f"  ·  {np.linalg.norm(c[:, -1, :].std(axis=0)):.3f} m"

        ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
        ax.set_xlabel("x (m)", labelpad=6); ax.set_ylabel("y (m)", labelpad=6)
        ax.set_zlabel("z (m)", labelpad=6)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=22, azim=-58)
        # Subtle, light 3D panes for a cleaner look. mplot3d internals shift
        # between matplotlib versions, so never let styling break the figure.
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            try:
                axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                axis.pane.set_edgecolor("#dddddd")
                axis._axinfo["grid"].update(color="#ececec", linewidth=0.7)
            except Exception:  # noqa: BLE001 - cosmetic only
                pass

        fig.suptitle(f"End-effector spread — {item}", y=0.97)
        ax.set_title("forward kinematics of sampled joint chunks · "
                     "legend shows end-point std",
                     fontsize=9, color=_MUTED, pad=2)
        _prompt_legend(fig, labels, colors, training_labels, extra=spreads)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        out = save_dir / f"end_effector_{_safe_name(item)}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  Saved: {out}")
