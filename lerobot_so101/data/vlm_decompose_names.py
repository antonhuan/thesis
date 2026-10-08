"""
Which names the perception output used for each object, one-pass vs two-pass.

For each case the perception output is the single call's visible_objects (one-pass) or
the identify pass's object list (two-pass). Each name is mapped to banana / toy / pouch
the way the decompose test scores "objects identified": the synonym list, then vague
names ("brown object") by elimination. A case with no name for an object counts as
"(not listed)".

Usage (inside the lerobot-so101 container, cwd /workspace):
    python vlm_decompose_names.py \\
        --a vlm_decompose_test/2026-10-02_07-42-36/results.json --a-name "one pass" \\
        --b vlm_decompose_test/2026-10-02_07-42-16/results.json --b-name "two pass" \\
        --out vlm_decompose_test/object_names.png
"""

import argparse
import collections
import json
from pathlib import Path

from vlm_decompose_test import OBJECTS, canonical, elimination_map

NOT_LISTED = "(not listed)"


def name_counts(path: str) -> tuple[dict[str, collections.Counter], int]:
    trials = json.loads(Path(path).read_text())["trials"]
    counts = {o: collections.Counter() for o in OBJECTS}
    for t in trials:
        names = list(dict.fromkeys(str(n).strip().lower() for n in t["identified"]))
        elim = elimination_map(names)
        seen = set()
        for n in names:
            obj = canonical(n) or elim.get(n)
            if obj:
                counts[obj][n] += 1
                seen.add(obj)
        for obj in set(OBJECTS) - seen:
            counts[obj][NOT_LISTED] += 1
    return counts, len(trials)


def plot(a, b, na, nb, objects, n, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Same ink, surface and series colours as vlm_decompose_compare.py.
    TEXT, TEXT2, GRID, BG = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    CA, CB = "#2a78d6", "#eb6834"
    plt.rcParams.update({"font.size": 9, "text.color": TEXT, "axes.labelcolor": TEXT2,
                         "xtick.color": TEXT2, "ytick.color": TEXT})
    rows = {o: sorted((set(a[o]) | set(b[o])) - {NOT_LISTED},
                      key=lambda k: -(a[o][k] + b[o][k])) + [NOT_LISTED] for o in objects}
    sizes = [len(rows[o]) for o in objects]
    # Vertical bars: each panel is as wide as its list of names, so bars match in width.
    fig, axes = plt.subplots(1, len(objects), facecolor=BG, sharey=True,
                             figsize=(0.55 * sum(sizes) + 1.5, 4.6),
                             gridspec_kw=dict(width_ratios=sizes, wspace=0.08))
    axes = [axes] if len(objects) == 1 else list(axes)
    w = 0.38
    ymax = max(max(max(c[o].values()) for o in objects) for c in (a, b))
    for ax, o in zip(axes, objects):
        names = rows[o]
        x = range(len(names))
        va, vb = [a[o][k] for k in names], [b[o][k] for k in names]
        ax.bar([i - w / 2 - 0.01 for i in x], va, width=w, color=CA, label=na)
        ax.bar([i + w / 2 + 0.01 for i in x], vb, width=w, color=CB, label=nb)
        for i, (y0, y1) in enumerate(zip(va, vb)):
            if y0:
                ax.text(i - w / 2 - 0.01, y0 + ymax * 0.01, str(y0), ha="center", va="bottom", fontsize=7)
            if y1:
                ax.text(i + w / 2 + 0.01, y1 + ymax * 0.01, str(y1), ha="center", va="bottom", fontsize=7)
        ax.set_xticks(list(x), names, rotation=45, ha="right", rotation_mode="anchor")
        ax.get_xticklabels()[-1].set_color(TEXT2)
        ax.get_xticklabels()[-1].set_style("italic")
        ax.axvline(len(names) - 1.5, color=GRID, lw=0.8)  # separates the misses
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_ylim(0, ymax * 1.1)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ["top", "right"] + (["left"] if ax is not axes[0] else []):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.spines["left"].set_color(GRID)
        ax.tick_params(length=0)
        ax.set_title(o, loc="left", fontsize=10)
    axes[0].set_ylabel(f"cases (of {n})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24),
               ncol=2, fontsize=9)
    fig.suptitle("Object names used by the perception step", fontsize=12)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Object-name distribution, one-pass vs two-pass")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-name", default="run A")
    ap.add_argument("--b-name", default="run B")
    ap.add_argument("--objects", default="toy,pouch", help=f"Comma list from {OBJECTS}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    (a, na_n), (b, nb_n) = name_counts(args.a), name_counts(args.b)
    objects = args.objects.split(",")
    plot(a, b, args.a_name, args.b_name, objects, na_n, args.out)
    for o in OBJECTS:
        print(f"{o}: {args.a_name} {dict(a[o].most_common())} | {args.b_name} {dict(b[o].most_common())}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
