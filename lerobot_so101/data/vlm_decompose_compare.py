"""
Side-by-side comparison of two vlm_decompose_test runs on the same scene x prompt pairs
(e.g. the original one-pass planner vs the two-pass planner).

Only pairs present in both runs are compared, so a --sample run and its matching
subset line up case for case.

Usage (inside the lerobot-so101 container, cwd /workspace):
    python vlm_decompose_compare.py \\
        --a vlm_decompose_test/2026-10-01_02-52-25/results.json --a-name "one-pass (original)" \\
        --b vlm_decompose_test/2026-09-27_03-43-57_sample200/results.json --b-name "two-pass" \\
        --out vlm_decompose_test/compare_one_vs_two_pass

Outputs (--out): summary.md, comparison.png
"""

import argparse
import collections
import json
from pathlib import Path

import math

from vlm_decompose_test import (CATEGORIES, CHECKS, OBJECTS, PROMPTS, canonical, effective_correct,
                                elimination_map, normalise)

CHECK_NAMES = {"identified": "all three objects identified", "object_set": "right objects planned",
               "order": "order respected", "count": "right number of sub-tasks",
               "destination": "destination is the tray", "correct": "overall correct"}


def load(path: str) -> dict:
    results = json.loads(Path(path).read_text())
    normalise(results)
    return {(t["scene"], t["key"]): t for t in results["trials"] if t["trial"] == 0}


def rate(trials: list[dict], check: str) -> float:
    if check == "correct":
        return sum(effective_correct(t)[0] for t in trials) / len(trials)
    return sum(bool(t["auto_label"][check]) for t in trials) / len(trials)


def passed(t: dict, check: str) -> bool:
    return effective_correct(t)[0] if check == "correct" else bool(t["auto_label"][check])


def mcnemar(a_ok: list[bool], b_ok: list[bool]) -> float:
    """Exact two-sided McNemar test: binomial on the pairs where exactly one passes."""
    a_only = sum(x and not y for x, y in zip(a_ok, b_ok))
    b_only = sum(y and not x for x, y in zip(a_ok, b_ok))
    n = a_only + b_only
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2 ** n) if n else 1.0


def compare(a: dict, b: dict) -> dict:
    keys = sorted(set(a) & set(b))
    A, B = [a[k] for k in keys], [b[k] for k in keys]
    checks = {c: (rate(A, c), rate(B, c), mcnemar([passed(t, c) for t in A], [passed(t, c) for t in B]))
              for c in CHECKS + ["correct"]}
    cats = {}
    for cat in CATEGORIES:
        idx = [i for i, k in enumerate(keys) if PROMPTS[k[1]]["category"] == cat]
        if idx:
            cats[cat] = (rate([A[i] for i in idx], "correct"), rate([B[i] for i in idx], "correct"), len(idx))
    prompts = {}
    for key in PROMPTS:
        idx = [i for i, k in enumerate(keys) if k[1] == key]
        if idx:
            prompts[key] = (rate([A[i] for i in idx], "correct"), rate([B[i] for i in idx], "correct"), len(idx))
    paired = collections.Counter((effective_correct(x)[0], effective_correct(y)[0]) for x, y in zip(A, B))
    modes = {n: collections.Counter(t["auto_label"]["failure_mode"] for t in T if not effective_correct(t)[0])
             for n, T in (("a", A), ("b", B))}
    def seen(t):  # as the "identified" check: synonyms, then vague names by elimination
        ids = [str(x) for x in t["identified"]]
        return {canonical(x) for x in ids} | set(elimination_map(ids).values())
    missed = {n: collections.Counter(o for t in T for o in OBJECTS if o not in seen(t))
              for n, T in (("a", A), ("b", B))}
    # Does perception depend on the instruction? All-three-identified rate, split by
    # whether the instruction names each object.
    named = {n: {o: tuple((sum(bool(t["auto_label"]["identified"]) for t in S), len(S))
                          for S in ([t for t in T if o in t["prompt"].lower()],
                                    [t for t in T if o not in t["prompt"].lower()]))
                 for o in OBJECTS}
             for n, T in (("a", A), ("b", B))}
    return dict(n=len(keys), checks=checks, cats=cats, prompts=prompts, paired=paired,
                modes=modes, missed=missed, named=named)


def write_summary(c: dict, na: str, nb: str, args, path: Path):
    p = c["paired"]
    lines = [f"# Decomposition: {na} vs {nb}", "",
             f"- {na}: `{args.a}`", f"- {nb}: `{args.b}`",
             f"- {c['n']} scene x prompt pairs present in both runs, one trial each", "",
             "## Checks", "", f"| check | {na} | {nb} | change | p (exact McNemar) |",
             "|---|---|---|---|---|"]
    lines += [f"| {CHECK_NAMES[k]} | {x:.2f} | {y:.2f} | {y - x:+.2f} | {pv:.2g} |"
              for k, (x, y, pv) in c["checks"].items()]
    lines += ["", "## Overall correct by category", "", f"| category | n | {na} | {nb} | change |",
              "|---|---|---|---|---|"]
    lines += [f"| {k} | {n} | {x:.2f} | {y:.2f} | {y - x:+.2f} |" for k, (x, y, n) in c["cats"].items()]
    b_only, a_only = p[(False, True)], p[(True, False)]
    k, n = min(a_only, b_only), a_only + b_only  # exact two-sided McNemar (binomial, p = 0.5)
    pval = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    lines += ["", "## Case by case (same pairs)", "",
              f"Exact McNemar test on the {n} pairs where the two disagree: p = {pval:.3f}.", "",
              "| outcome | pairs |", "|---|---|",
              f"| both correct | {p[(True, True)]} |", f"| only {nb} correct | {p[(False, True)]} |",
              f"| only {na} correct | {p[(True, False)]} |", f"| both wrong | {p[(False, False)]} |",
              "", "## Failure modes (wrong cases)", "", f"| failure mode | {na} | {nb} |", "|---|---|---|"]
    for m in sorted(set(c["modes"]["a"]) | set(c["modes"]["b"]), key=str):
        lines.append(f"| {m} | {c['modes']['a'][m]} | {c['modes']['b'][m]} |")
    lines += ["", "## Objects missed by the perception step (cases)", "", f"| object | {na} | {nb} |",
              "|---|---|---|"]
    lines += [f"| {o} | {c['missed']['a'][o]} | {c['missed']['b'][o]} |" for o in OBJECTS]
    lines += ["", "## All three objects identified, by whether the instruction names the object", "",
              f"| object | {na} named | {na} not named | {nb} named | {nb} not named |",
              "|---|---|---|---|---|"]
    fmt = lambda k, n: f"{k / n:.2f} ({k}/{n})" if n else "-"
    for o in OBJECTS:
        (ak, an), (ak2, an2) = c["named"]["a"][o]
        (bk, bn), (bk2, bn2) = c["named"]["b"][o]
        lines.append(f"| {o} | {fmt(ak, an)} | {fmt(ak2, an2)} | {fmt(bk, bn)} | {fmt(bk2, bn2)} |")
    lines += ["", "## Overall correct by prompt", "", f"| category | prompt | n | {na} | {nb} | change |",
              "|---|---|---|---|---|---|"]
    lines += [f"| {PROMPTS[k]['category']} | {PROMPTS[k]['prompt']} | {n} | {x:.2f} | {y:.2f} | {y - x:+.2f} |"
              for k, (x, y, n) in c["prompts"].items()]
    path.write_text("\n".join(lines) + "\n")


# Same ink, surface and categorical order as vlm_decompose_test's figures.
TEXT, TEXT2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CA, CB = "#2a78d6", "#eb6834"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "text.color": TEXT, "axes.labelcolor": TEXT2,
                         "xtick.color": TEXT2, "ytick.color": TEXT})
    return plt


def _panels(c: dict) -> dict:
    """The two panels: (labels, (rate A, rate B) pairs, title)."""
    ks, cs = list(c["checks"]), list(c["cats"])
    return {
        "plan_detail": ([CHECK_NAMES[k] for k in ks], [c["checks"][k][:2] for k in ks],
                        "Plan detail correctness"),
        "overall_by_category": ([f"{k} (n={c['cats'][k][2]})" for k in cs], [c["cats"][k][:2] for k in cs],
                                "Overall correct by preference category"),
    }


def _bars(axis, labels, pairs, title, na, nb, fs=9, title_kw=None):
    h = 0.36
    y = list(range(len(labels)))
    axis.barh([i - h / 2 - 0.01 for i in y], [p[0] for p in pairs], height=h, color=CA, label=na)
    axis.barh([i + h / 2 + 0.01 for i in y], [p[1] for p in pairs], height=h, color=CB, label=nb)
    for i, (x0, x1) in enumerate(pairs):
        axis.text(x0 + 0.015, i - h / 2 - 0.01, f"{x0:.0%}", va="center", fontsize=fs - 1, color=TEXT)
        axis.text(x1 + 0.015, i + h / 2 + 0.01, f"{x1:.0%}", va="center", fontsize=fs - 1, color=TEXT)
    axis.set_yticks(y, labels, fontsize=fs)
    axis.invert_yaxis()
    axis.set_xlim(0, 1.12)
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"], fontsize=fs)
    axis.grid(axis="x", color=GRID, lw=0.8)
    axis.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        axis.spines[s].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(length=0)
    axis.set_title(title, **(title_kw or dict(loc="left", fontsize=fs + 1)))


def plot(c: dict, na: str, nb: str, path: Path):
    """Both panels side by side in one figure."""
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=BG,
                             gridspec_kw=dict(wspace=0.55, width_ratios=[1.1, 1]))
    for axis, panel in zip(axes, _panels(c).values()):
        _bars(axis, *panel, na, nb)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0),
               ncol=2, fontsize=9)
    fig.suptitle(f"Decomposition: {na} vs {nb}", fontsize=12)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def plot_split(c: dict, na: str, nb: str, out: Path) -> list[Path]:
    """Each panel as its own figure, with its own legend, for separate placement."""
    plt = _plt()
    paths = []
    for name, panel in _panels(c).items():
        fig, axis = plt.subplots(figsize=(8, 5.6), facecolor=BG)
        labels, pairs, title = panel
        _bars(axis, labels, pairs, "", na, nb, fs=13)
        fig.legend(*axis.get_legend_handles_labels(), frameon=False, loc="upper center",
                   bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=13)
        # Centre the title over the saved (tight-cropped) image, row labels included,
        # rather than over the axes alone.
        bb = fig.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans)
        x = fig.transFigure.inverted().transform(((bb.x0 + bb.x1) / 2, 0))[0]
        fig.text(x, 0.93, title, ha="center", va="bottom", fontsize=15, fontweight="bold")
        path = out / f"{name}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        paths.append(path)
    return paths


def main():
    ap = argparse.ArgumentParser(description="Compare two vlm_decompose_test runs case for case")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-name", default="run A")
    ap.add_argument("--b-name", default="run B")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    c = compare(load(args.a), load(args.b))
    write_summary(c, args.a_name, args.b_name, args, out / "summary.md")
    plot(c, args.a_name, args.b_name, out / "comparison.png")
    split = plot_split(c, args.a_name, args.b_name, out)
    print(f"Wrote {out / 'summary.md'}, {out / 'comparison.png'}, "
          f"{', '.join(p.name for p in split)} ({c['n']} shared pairs)")


if __name__ == "__main__":
    main()
