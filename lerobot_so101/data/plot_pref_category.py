"""Per-category bar chart: VLA baseline vs. full system (replan off) vs. human oracle
(vla_preference_oracle_live.py), preference prompts only.

Task success rate per category plus an overall group, each bar labelled k/n. Reads the paired trials
exactly as the write-up does:
    python plot_pref_category.py
Writes preference_eval_system_trials/fig_pref_category.{pdf,png}. Sized for a 6.3 in
\\textwidth; include the PDF at width=\\textwidth so the text stays at 8-9 pt.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from vla_preference_eval_live import load_trials

CATS = [("omission", "Omission"), ("sequencing", "Sequencing"),
        ("conditional", "Conditional"), ("quantity", "Quantity")]
SERIES = [("Baseline (VLA only)", "#a3a3a3"), ("Full system", "#2a78d6"),
          ("Human oracle", "#eb6834")]
INK, INK2, GRID = "#1a1a1a", "#4a4a4a", "#e2e2e2"
OUT = Path("preference_eval_system_trials")

system = [t for t in load_trials(OUT) if t.get("label_ok")]
# "Put the yellow object on the tray": the planner selected all three objects and the
# operator aborted after the banana was placed. Scored as failed, as in the thesis.
ABORTED_AS_FAILED = {54, 58}
for t in system:
    if t["trial"] in ABORTED_AS_FAILED:
        t["scores"].update(complete=False, violated=True,
                           intent_complete=False, intent_violated=True)
pool = [t for t in load_trials(Path("preference_eval_trials")) if t.get("label_ok")]
pairs = []
for t in system:
    b = next(x for x in pool if (x["key"], x.get("layout")) == (t["key"], t.get("layout")))
    pool.remove(b)
    pairs.append((b, t))
pairs = [(b, s) for b, s in pairs if s["category"] in dict(CATS)]   # preference prompts only
# The oracle ran only the preference items of the same queue: pair it by (key, layout) too.
pool = [t for t in load_trials(Path("preference_eval_oracle_trials")) if t.get("label_ok")]
triples = []
for b, s in pairs:
    o = next(x for x in pool if (x["key"], x.get("layout")) == (s["key"], s.get("layout")))
    pool.remove(o)
    triples.append((b, s, o))


def counts(metric):
    """[(label, n, [baseline k, system k, oracle k])] per category, then overall."""
    rows = []
    for cat, label in CATS:
        ts = [tr for tr in triples if tr[1]["category"] == cat]
        rows.append((label, len(ts), [sum(bool(tr[j]["scores"][metric]) for tr in ts)
                                      for j in range(len(SERIES))]))
    rows.append(("Overall", sum(r[1] for r in rows),
                 [sum(r[2][j] for r in rows) for j in range(len(SERIES))]))
    return rows


plt.rcParams.update({
    "font.family": "serif", "font.serif": ["CMU Serif", "Computer Modern Roman",
                                           "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 12, "axes.unicode_minus": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK2,
})
metric = "complete"
rows = counts(metric)

fig, ax = plt.subplots(figsize=(6.3, 3.6))
w, gap, sep = 0.26, 0.06, 0.35         # bar width, gap within a group, extra gap before Overall
xs = [i + (sep if label == "Overall" else 0) for i, (label, *_) in enumerate(rows)]

for x0, (label, n, ks) in zip(xs, rows):
    for j, k in enumerate(ks):
        x = x0 + (j - (len(ks) - 1) / 2) * (w + gap)
        pct = 100 * k / n
        ax.bar(x, pct, w, color=SERIES[j][1], edgecolor="none", zorder=2)
        ax.text(x, pct + 2, f"{k}/{n}", ha="center", va="bottom", fontsize=8, color=INK)

# Divider before the overall group.
ax.axvline((xs[-2] + xs[-1]) / 2, color=GRID, linewidth=0.8, zorder=1)

ax.set_xticks(xs)
ax.set_xticklabels([f"{label}\n" + rf"($n={n}$)" for label, n, *_ in rows], fontsize=11)
ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
ax.set_ylim(0, 110)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("Task success rate (%)")
ax.set_title("Task success on preference prompts",
             fontsize=14, color=INK, pad=10)
ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis="x", length=0)
ax.tick_params(axis="y", length=2.5, width=0.6)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_linewidth(0.6)
ax.spines["left"].set_bounds(0, 100)

handles = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none") for _, c in SERIES]
fig.legend(handles, [s for s, _ in SERIES], loc="lower center", ncol=len(SERIES), frameon=False,
           bbox_to_anchor=(0.5, 0.0), fontsize=10, handlelength=1.2, handleheight=0.9,
           columnspacing=1.5)

fig.tight_layout(pad=0.3, rect=(0, 0.1, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_pref_category.{ext}", dpi=300, facecolor="white")
print(metric, rows)
