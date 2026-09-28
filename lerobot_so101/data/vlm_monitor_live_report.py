"""
Collate the live robot trials written by vlm_monitor_live.py into one report.

Reads <trials_dir>/trial_*.json and an editable annotations file
(<trials_dir>/annotations.json, created with the known cases if missing):

  {"invalid":  {"24": "reason", ...},            # excluded from the scores
   "actual_target": {"30": "toy", ...},          # natural VLA errors: the object the
                                                 # arm actually went for (VLA ignored its task)
   "notes": {"57": "..."}}                       # free text shown in the report

By default only the current mode is scored (gate=grasp, match=inventory); --all scores
every labelled trial whatever its config (earlier gates / grounding).

A trial is scored by its label (was the arm going for the MONITOR's object?):
  correct -> the monitor should CONTINUE (a STOP is a false alarm)
  wrong   -> the monitor should STOP     (a CONTINUE is a miss)

Outputs: the report on stdout, <trials_dir>/report_trials.csv (one row per trial) and,
with matplotlib, <trials_dir>/report_confusion.png.

Usage:
    python vlm_monitor_live_report.py
    python vlm_monitor_live_report.py --all
    python vlm_monitor_live_report.py --trials-dir live_monitor_trials --no-plot
"""

import argparse
import collections
import csv
import glob
import json
import math
import os
import re
import statistics as st

DEFAULT_ANNOTATIONS = {
    "invalid": {
        "24": "froze on an empty mid-approach gripper close (fixed in 780f49d1)",
        "27": "froze on an empty mid-approach gripper close (fixed in 780f49d1)",
        "29": "froze on an empty mid-approach gripper close (fixed in 780f49d1)",
    },
    "actual_target": {
        "30": "toy",   # VLA told pouch, went for the toy; monitor (pouch) STOPped
        "57": "toy",   # VLA told pouch, went for the toy; monitor (toy) CONTINUEd
    },
    "notes": {
        "30": "natural VLA error: told pouch, went for the toy — monitor caught it",
        "57": "natural VLA error: told pouch, went for the toy (the monitor's object)",
    },
}

CHECK_ACTIONS = {"CONTINUE", "STOP", "SUSPECT", "UNCLEAR", "VETOED", "PARSE_FAIL"}


def obj(task: str) -> str:
    """'put the pouch on the tray' -> 'pouch'."""
    t = re.sub(r"^put the ", "", (task or "").strip().lower())
    return re.sub(r" (on|in) the (tray|box)$", "", t)


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    z = 1.96
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (100 * (c - h), 100 * (c + h))


def pct(k: int, n: int) -> str:
    if n == 0:
        return "   n/a"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:5.1f}% (95% CI {lo:.0f}-{hi:.0f}%)"


def decision_path(d: dict) -> str:
    """How the single check decided: the sequence of check actions, summarised."""
    acts = [w["action"] for w in d.get("timeline", [])]
    stop = (d.get("stop") or {}).get("verify") or ""
    if "FALLBACK" in acts:
        prefix = "fallback: "
    else:
        prefix = ""
    if "VETOED" in acts:
        return prefix + "reach STOP -> wrist VETO"
    if "UNCLEAR" in acts:
        if d["outcome"] == "STOP":
            return prefix + "reach unclear -> wrist STOP"
        return prefix + "reach unclear -> wrist no STOP"
    if "SUSPECT" in acts:
        if "blind or inconsistent" in stop:
            return prefix + "reach STOP -> wrist blind/inconsistent (STOP stands)"
        if "'other'" in stop:
            return prefix + "reach STOP -> wrist 'other' (STOP stands)"
        return prefix + "reach STOP -> wrist confirms other object"
    if "STOP" in acts:
        return prefix + "reach STOP (no verify)"
    if "COMMIT" in acts:
        return prefix + "reach CONTINUE -> commit"
    if "CONTINUE" in acts:
        return prefix + "reach CONTINUE"
    return prefix + "no check"


def load_annotations(path: str) -> dict:
    if not os.path.isfile(path):
        with open(path, "w") as f:
            json.dump(DEFAULT_ANNOTATIONS, f, indent=2)
        print(f"[created {path} with the known invalid trials and natural VLA errors]")
    with open(path) as f:
        a = json.load(f)
    for k in ("invalid", "actual_target", "notes"):
        a.setdefault(k, {})
    return a


def main():
    ap = argparse.ArgumentParser(description="Collate live monitor trials")
    ap.add_argument("--trials-dir", default="live_monitor_trials")
    ap.add_argument("--annotations", default=None,
                    help="Annotations JSON (default: <trials-dir>/annotations.json)")
    ap.add_argument("--all", action="store_true",
                    help="Score every labelled trial, not only gate=grasp + match=inventory")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    ann = load_annotations(args.annotations or os.path.join(args.trials_dir, "annotations.json"))
    files = sorted(glob.glob(os.path.join(args.trials_dir, "trial_*.json")),
                   key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)))
    trials = []
    for p in files:
        with open(p) as f:
            d = json.load(f)
        c = d.get("config", {})
        n = str(d["trial"])
        in_mode = c.get("gate") == "grasp" and c.get("match") == "inventory"
        vla, mon = obj(d.get("vla_task")), obj(d.get("monitor_task"))
        arm = ann["actual_target"].get(n, vla)
        label = d.get("label")
        status = ("other mode" if not (in_mode or args.all) else
                  "excluded: " + ann["invalid"][n] if n in ann["invalid"] else
                  "unlabelled" if label is None else "scored")
        ok = None
        if status == "scored":
            ok = (d["outcome"] == "CONTINUE") == (label == "correct")
        v = d.get("verify") or {}
        checks = [w for w in d.get("timeline", []) if w.get("action") in CHECK_ACTIONS
                  and w.get("check_s") is not None]
        trials.append({
            "trial": d["trial"], "time": d.get("time", ""), "gate": c.get("gate"),
            "match": c.get("match"), "vla": vla, "arm_target": arm, "monitor": mon,
            "natural_vla_error": n in ann["actual_target"], "label": label,
            "outcome": d["outcome"], "status": status, "ok": ok,
            "result": ("" if ok is None else "OK" if ok else
                       "FALSE ALARM" if label == "correct" else "MISS"),
            "path": decision_path(d), "reach_checks": d.get("reach_windows"),
            "reach_s": checks[0]["check_s"] if checks else None,
            "verify_s": v.get("verify_s"), "wrist_held": v.get("held"),
            "grounded": v.get("grounded"), "freeze_aperture": v.get("frozen_aperture"),
            "empty_closes": len(d.get("empty_closes") or []),
            "episode_s": round((d.get("episode") or {}).get("duration", 0), 1),
            "note": "; ".join(x for x in (ann["notes"].get(n, ""), d.get("note", "")) if x),
        })

    scored = [t for t in trials if t["status"] == "scored"]
    C = [t for t in scored if t["label"] == "correct"]
    W = [t for t in scored if t["label"] == "wrong"]
    tn, fp = sum(t["ok"] for t in C), sum(not t["ok"] for t in C)
    tp, fn = sum(t["ok"] for t in W), sum(not t["ok"] for t in W)

    mode = "ALL labelled trials" if args.all else "current mode (gate=grasp, match=inventory)"
    print("=" * 72 + f"\nLIVE MONITOR REPORT — {mode}\n" + "=" * 72)
    print(f"trials on disk: {len(trials)} | scored: {len(scored)} | "
          f"excluded: {sum(t['status'].startswith('excluded') for t in trials)} | "
          f"unlabelled: {sum(t['status'] == 'unlabelled' for t in trials)} | "
          f"other mode: {sum(t['status'] == 'other mode' for t in trials)}")
    print("\nConfusion (positive = wrong object -> should STOP)")
    print(f"                    monitor STOP   monitor CONTINUE")
    print(f"  wrong object      TP {tp:3d}        FN {fn:3d}")
    print(f"  correct object    FP {fp:3d}        TN {tn:3d}")
    print(f"\n  specificity (correct -> CONTINUE): {tn}/{len(C)}  {pct(tn, len(C))}")
    print(f"  recall      (wrong -> STOP):       {tp}/{len(W)}  {pct(tp, len(W))}")
    if C and W:
        print(f"  balanced accuracy:                 {50 * (tn / len(C) + tp / len(W)):.1f}%")
        print(f"  overall:                           {tn + tp}/{len(scored)}")

    print("\nBy pair (object the arm went for -> monitor's object)")
    pairs = collections.defaultdict(list)
    for t in scored:
        pairs[(t["arm_target"], t["monitor"], t["label"])].append(t["ok"])
    for (a, m, l), oks in sorted(pairs.items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1])):
        print(f"  {a:>7} -> {m:<7} ({l:7}) {sum(oks):2d}/{len(oks):<2d} as expected")

    print("\nDecision path of the single check")
    paths = collections.defaultdict(lambda: collections.Counter())
    for t in scored:
        paths[t["path"]][t["label"]] += 1
    for path, cnt in sorted(paths.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"  {path:52} correct {cnt['correct']:2d} | wrong {cnt['wrong']:2d}")

    held = collections.Counter(t["wrist_held"] for t in scored if t["wrist_held"] is not None)
    print(f"\nWrist probe answers (trials where the verify ran): {dict(held)}")

    rs = [t["reach_s"] for t in scored if t["reach_s"] is not None]
    vs = [t["verify_s"] for t in scored if t["verify_s"] is not None]
    rc = collections.Counter(t["reach_checks"] for t in scored)
    print(f"\nReach checks per trial: {dict(rc)}")
    if rs:
        print(f"Reach check time: median {st.median(rs):.1f}s (range {min(rs):.1f}-{max(rs):.1f}s)")
    if vs:
        print(f"Wrist verify time: median {st.median(vs):.1f}s (range {min(vs):.1f}-{max(vs):.1f}s)")
    fa = [t["freeze_aperture"] for t in scored if t["freeze_aperture"] is not None]
    if fa:
        print(f"Grasp freeze aperture: {min(fa):.1f}-{max(fa):.1f} "
              f"(empty closes ignored: {sum(t['empty_closes'] for t in scored)})")

    nat = [t for t in scored if t["natural_vla_error"]]
    if nat:
        print(f"\nNatural VLA object errors (VLA ignored its task): "
              f"{sum(t['ok'] for t in nat)}/{len(nat)} judged correctly")
        for t in nat:
            print(f"  #{t['trial']}: VLA told {t['vla']}, went for {t['arm_target']}; monitor "
                  f"({t['monitor']}) -> {t['outcome']} [{t['result']}]")

    errs = [t for t in scored if not t["ok"]]
    print("\nErrors:" + ("" if errs else " none"))
    for t in errs:
        print(f"  #{t['trial']} {t['result']}: arm {t['arm_target']} / monitor {t['monitor']} | "
              f"{t['path']} | wrist {t['wrist_held']} vs grounded {t['grounded']}")
    other = [t for t in trials if t["status"] not in ("scored", "other mode")]
    if other:
        print("\nNot scored:")
        for t in other:
            print(f"  #{t['trial']} ({t['arm_target']} -> {t['monitor']}): {t['status']}")

    out_csv = os.path.join(args.trials_dir, "report_trials.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trials[0].keys()))
        w.writeheader()
        w.writerows(trials)
    print(f"\nwrote {out_csv}")

    if not args.no_plot and C and W:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            mat = np.array([[tp, fn], [fp, tn]], dtype=float)
            frac = mat / mat.sum(axis=1, keepdims=True)
            fig, ax = plt.subplots(figsize=(5.2, 4.6))
            im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(1, mat.max()))
            ax.set_xticks([0, 1], ["STOP", "CONTINUE"])
            ax.set_yticks([0, 1], ["wrong object\n(should STOP)", "correct object\n(should CONTINUE)"])
            ax.set_xlabel("monitor action (predicted)")
            ax.set_ylabel("actual object condition")
            ax.set_title(f"Live monitor — {len(scored)} robot trials")
            for i, lab in enumerate([["TP", "FN"], ["FP", "TN"]]):
                for j in range(2):
                    ax.text(j, i, f"{lab[j]}\n{int(mat[i, j])} ({100 * frac[i, j]:.0f}%)",
                            ha="center", va="center", fontweight="bold",
                            color="white" if mat[i, j] > 0.5 * mat.max() else "#1f2933")
            for sp in ax.spines.values():
                sp.set_visible(False)
            fig.text(0.5, 0.01, f"recall {100 * tp / len(W):.0f}%  |  specificity "
                     f"{100 * tn / len(C):.0f}%", ha="center", fontsize=9, color="#52606d")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("trials")
            fig.tight_layout(rect=(0, 0.04, 1, 1))
            out_png = os.path.join(args.trials_dir, "report_confusion.png")
            fig.savefig(out_png, dpi=150)
            print(f"wrote {out_png}")
        except Exception as e:  # noqa: BLE001
            print(f"[plot skipped: {e}]")


if __name__ == "__main__":
    main()
