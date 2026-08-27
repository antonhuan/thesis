"""
Patch script: applies the joint-AUC metric additions to vla_denoise_consistency.py.
Run once, then delete this file.
"""
import re

path = __import__("os").path.join(
    __import__("os").path.dirname(__import__("os").path.abspath(__file__)),
    "vla_denoise_consistency.py",
)

with open(path) as f:
    src = f.read()

# --------------------------------------------------------------------------- #
# 1. Add PLANNING_JOINTS constant + compute_joint_auc after NO_MOVEMENT_THRESHOLD
# --------------------------------------------------------------------------- #
new_constant_and_func = '''NO_MOVEMENT_THRESHOLD = 10.0

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

'''

src = src.replace("NO_MOVEMENT_THRESHOLD = 10.0\n", new_constant_and_func, 1)

# --------------------------------------------------------------------------- #
# 2. In run_episode, compute joint_auc per label after the variance curve and
#    inject it into the record's metrics dict.
# --------------------------------------------------------------------------- #
# Find the line that stores the variance curve and add the AUC right after.
old_var_line = '''                record_curves[label] = np.asarray(real, dtype=float).var(axis=0)

                elapsed = time.time() - t0'''

new_var_line = '''                record_curves[label] = np.asarray(real, dtype=float).var(axis=0)
                metrics["joint_auc"] = compute_joint_auc(record_curves[label])
                metrics["joint_auc_late"] = compute_joint_auc(
                    record_curves[label], t_start=record_curves[label].shape[0] // 2)

                elapsed = time.time() - t0'''

src = src.replace(old_var_line, new_var_line, 1)

# --------------------------------------------------------------------------- #
# 3. Add joint_auc and joint_auc_late to the metric_names tuple in
#    aggregate_results so they get aggregated across records.
# --------------------------------------------------------------------------- #
old_metric_names = '''    metric_names = ("mean_std", "max_std", "mean_cv", "max_cv",
                    "action_magnitude", "ee_consistency", "peak_disp_mean",
                    "no_move_frac", "gt_dist_mean")'''

new_metric_names = '''    metric_names = ("mean_std", "max_std", "mean_cv", "max_cv",
                    "action_magnitude", "ee_consistency", "peak_disp_mean",
                    "no_move_frac", "gt_dist_mean",
                    "joint_auc", "joint_auc_late")'''

src = src.replace(old_metric_names, new_metric_names)

# --------------------------------------------------------------------------- #
# 4. Add Joint AUC columns to format_results_table
# --------------------------------------------------------------------------- #
old_per_rec_header = '''              f"{'No-Move %':>10} {'GT Dist':>10}")'''
new_per_rec_header = '''              f"{'No-Move %':>10} {'GT Dist':>10} "
              f"{'Jnt AUC':>10} {'Jnt AUC Late':>12}")'''
src = src.replace(old_per_rec_header, new_per_rec_header, 1)

old_per_rec_row = '''            f"{_num(no_move_pct, 10, 1)} "
            f"{_num(data.get('gt_dist_mean'), 10, 4)}"'''
new_per_rec_row = '''            f"{_num(no_move_pct, 10, 1)} "
            f"{_num(data.get('gt_dist_mean'), 10, 4)} "
            f"{_num(data.get('joint_auc'), 10, 2)} "
            f"{_num(data.get('joint_auc_late'), 12, 2)}"'''
src = src.replace(old_per_rec_row, new_per_rec_row, 1)

# --------------------------------------------------------------------------- #
# 5. Add Joint AUC columns to format_aggregate_table
# --------------------------------------------------------------------------- #
old_agg_header = '''              f"{'No-Move % (mean+-sd)':>22} "
              f"{'GT Dist (mean+-sd)':>22}")'''
new_agg_header = '''              f"{'No-Move % (mean+-sd)':>22} "
              f"{'GT Dist (mean+-sd)':>22} "
              f"{'Jnt AUC (mean+-sd)':>22} "
              f"{'Jnt AUC Late (mean+-sd)':>26}")'''
src = src.replace(old_agg_header, new_agg_header, 1)

old_agg_row = '''            f"{cell(no_move_pct_m, d['no_move_frac_sd'] * 100.0, 1):>22} "
            f"{cell(d['gt_dist_mean_mean'], d['gt_dist_mean_sd'], 4):>22}"'''
new_agg_row = '''            f"{cell(no_move_pct_m, d['no_move_frac_sd'] * 100.0, 1):>22} "
            f"{cell(d['gt_dist_mean_mean'], d['gt_dist_mean_sd'], 4):>22} "
            f"{cell(d.get('joint_auc_mean', float('nan')), d.get('joint_auc_sd', 0.0), 2):>22} "
            f"{cell(d.get('joint_auc_late_mean', float('nan')), d.get('joint_auc_late_sd', 0.0), 2):>26}"'''
src = src.replace(old_agg_row, new_agg_row, 1)

# --------------------------------------------------------------------------- #
# 6. In write_object_aggregate, save the mean variance curves as .npy
# --------------------------------------------------------------------------- #
old_plot_call = '''    plot_joint_variance_over_time(
        {label: {} for label in mean_curves},
        entry["item_groups"], TRAINING_LABELS,
        out_dir / "joint_variance", unnorm=unnorm,
        variance_curves=mean_curves,
    )'''

new_plot_call = '''    # Save raw mean-variance curves so downstream analysis can compute
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
    )'''

src = src.replace(old_plot_call, new_plot_call, 1)

with open(path, "w") as f:
    f.write(src)

print("Patch applied successfully.")
