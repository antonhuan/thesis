"""
VLA baseline: preference instruction vs base instruction in PI05 embedding space.

The orchestrator never hands a preference instruction ("put everything on the
tray except for the banana") to the VLA directly -- the VLM decomposes it into
base sub-tasks the VLA was trained on ("put the toy on the tray", ...). This
script is the baseline for that design: it feeds the *raw* preference
instruction to the VLA and measures where it lands in the VLM backbone's final
hidden-state space relative to the base instructions.

Preferences follow the thesis taxonomy over the object set
X = {banana, toy, pouch} (base goal set G_b(z0) = X, one tray):

    omission       excluded set E_p;          admissible objects  G_b \\ E_p
    sequencing     strict partial order <_p;  admissible first    <_p-minimal objects
    conditional    scene predicate phi_p;     admissible objects  {x : phi_p(x, z0)}
    quantity       class c, count n;          admissible objects  c

Each preference therefore defines an admissible set A_p -- the objects the VLA
could legitimately act on next. A VLA that understood the preference would sit
closer to the base instructions of A_p than to those of X \\ A_p. The per-
preference score is

    violation pull = mean sim(pref, X \\ A_p) - mean sim(pref, A_p)

(> 0: the embedding leans toward an inadmissible object). It is undefined
when A_p = X (e.g. "put one object on the tray"); those preferences report only
their nearest base instruction.

All metrics are computed WITHIN a scene: every base and preference prompt is
encoded on the same frame + joint state, so the comparison between prompts is
free of scene-to-scene variation. Results are reported per scene; the
cross-scene view only counts how many scenes agree, it never mixes scenes into
one similarity.

Architecture target (same as vla_embedding_comp.py):
    PI05Policy.model.paligemma_with_expert.paligemma.model.language_model
    → forward() → .last_hidden_state  [B, seq_len, 2048]
    i.e. the representation cached as KV pairs for the action expert.

Inputs route through the checkpoint's real preprocessor (training-faithful
"Task: .., State: ..; Action:" prompt, front + wrist cameras).

Usage:
    # Default: timestep-0 frames of every episode under vla_inputs/
    python vla_preference_baseline.py

    # Only some scenes / a different root / checkpoint:
    python vla_preference_baseline.py --scenes 001_banana 012_toy
    python vla_preference_baseline.py --episode_root vla_failure_test \
        --max_scenes 20 --checkpoint ant0nh/pi05_500_30k

    # A single saved image (stateful prompt uses INITIAL_POSE) or live camera:
    python vla_preference_baseline.py --image_path scene.png
    python vla_preference_baseline.py --live

Outputs (in <output>/<run_stamp>/):
    results.json, summary.md
    consistency_<region>.png      scene x preference violation-pull grid
    t-SNE plots are organised by base instruction: each shows one base
    instruction's preference versions (VERSION_GROUPS) with all three base
    instructions as anchors, coloured by the base each version latches onto.
    tsne_scenes_<region>_<object>.png  per-scene maps for one base instruction
    tsne_<region>.png/.json            all-scene t-SNE, one panel per base instruction
    scenes/<scene>/tsne_<region>.png   one scene, one panel per base instruction
    scenes/<scene>/bars_<region>.png, heatmap_<region>.png
    scenes/<first scene>/per_token_<pref>.png
"""

import argparse
import json
import logging
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vla_embedding_utils import (
    build_preprocessor,
    extract_vlm_hidden_states,
    load_policy,
    load_scene_image,
    pool_embeddings,
    pool_region,
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

OBJECTS = ["banana", "toy", "pouch"]

# Base instructions: the training labels, i.e. what the VLM decomposes into.
BASE_PROMPTS = OrderedDict((o, f"put the {o} on the tray") for o in OBJECTS)

CATEGORIES = ("omission", "sequencing", "conditional", "quantity")


def _omission(prompt, excluded):
    return dict(category="omission", prompt=prompt, excluded=excluded,
                admissible=[o for o in OBJECTS if o not in excluded])


def _sequencing(prompt, order):
    """``order``: list of (before, after) pairs defining the strict partial
    order <_p. Admissible first picks are the minimal elements."""
    after = {b for _, b in order}
    return dict(category="sequencing", prompt=prompt, order=order,
                admissible=[o for o in OBJECTS if o not in after])


def _conditional(prompt, satisfies):
    return dict(category="conditional", prompt=prompt, admissible=satisfies)


def _quantity(prompt, cls, n):
    return dict(category="quantity", prompt=prompt, cls=cls, n=n, admissible=cls)


# Scene facts the conditional predicates rely on (all vla_inputs scenes):
# banana = yellow fruit; toy = brown furry plush moose; pouch = brown
# cylindrical pouch. There is a single tray, so the receptacle assignment h_p is
# trivially "tray" and is not varied here.
PREFERENCE_PROMPTS = OrderedDict([
    # --- Omission ---------------------------------------------------------
    ("omit banana",        _omission("put everything on the tray except the banana", ["banana"])),
    ("omit toy",           _omission("put everything on the tray except the toy", ["toy"])),
    ("omit pouch",         _omission("put everything on the tray except the pouch", ["pouch"])),
    ("omit banana+toy",    _omission("put everything on the tray except the banana and the toy",
                                     ["banana", "toy"])),
    ("omit banana+pouch",  _omission("put everything on the tray except the banana and the pouch",
                                     ["banana", "pouch"])),
    ("omit toy+pouch",     _omission("put everything on the tray except the toy and the pouch",
                                     ["toy", "pouch"])),
    # --- Sequencing -------------------------------------------------------
    ("seq banana>toy>pouch", _sequencing(
        "put the banana on the tray first, then the toy, then the pouch",
        [("banana", "toy"), ("toy", "pouch"), ("banana", "pouch")])),
    ("seq toy>pouch>banana", _sequencing(
        "put the toy on the tray first, then the pouch, then the banana",
        [("toy", "pouch"), ("pouch", "banana"), ("toy", "banana")])),
    ("seq pouch>banana>toy", _sequencing(
        "put the pouch on the tray first, then the banana, then the toy",
        [("pouch", "banana"), ("banana", "toy"), ("pouch", "toy")])),
    # Partial orders where the first-mentioned noun is NOT a minimal element.
    ("seq banana after toy", _sequencing(
        "put the banana on the tray after the toy", [("toy", "banana")])),
    ("seq pouch last",       _sequencing(
        "put the pouch on the tray last", [("banana", "pouch"), ("toy", "pouch")])),
    # --- Conditional selection -------------------------------------------
    ("cond yellow",        _conditional("put the yellow object on the tray", ["banana"])),
    ("cond fruit",         _conditional("put the fruit on the tray", ["banana"])),
    ("cond furry",         _conditional("put the furry object on the tray", ["toy"])),
    ("cond cylindrical",   _conditional("put the cylindrical object on the tray", ["pouch"])),
    ("cond brown",         _conditional("put the brown objects on the tray", ["toy", "pouch"])),
    ("cond not edible",    _conditional("put everything that is not edible on the tray",
                                        ["toy", "pouch"])),
    # --- Quantity ---------------------------------------------------------
    ("qty one object",     _quantity("put one object on the tray", list(OBJECTS), 1)),
    ("qty two objects",    _quantity("put two objects on the tray", list(OBJECTS), 2)),
    ("qty one brown",      _quantity("put one of the brown objects on the tray",
                                     ["toy", "pouch"], 1)),
    ("qty one fruit",      _quantity("put only one fruit on the tray", ["banana"], 1)),
])

# Initial joint pose for single-image / live modes (episode mode uses each
# frame's recorded state). Same rest pose as vla_denoise_consistency.py.
INITIAL_POSE = [-4.571428571428571, -101.49450549450549, 91.91208791208791,
                74.28571428571429, -0.7472527472527473, 1.3013698630136987]

REGIONS = ("full_prefix", "image_tokens", "language_tokens")

# Categorical slots 1-3 (validated all-pairs, see dataviz palette), fixed per
# object so colour follows the entity across every figure.
BASE_COLORS = {"banana": "#2a78d6", "toy": "#eb6834", "pouch": "#1baf7a"}
PREF_COLOR = "#52514e"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
# Diverging poles for violation pull: blue = leans admissible, orange = leans
# inadmissible, neutral grey midpoint.
DIVERGING = ["#2a78d6", "#f0efec", "#eb6834"]


# --------------------------------------------------------------------------- #
# Scene loading
# --------------------------------------------------------------------------- #

def load_episode_scenes(root, frame=0, max_scenes=None, only=None):
    """One scene per episode directory (any depth) under ``root`` that has a
    metadata.jsonl: the record at ``timestep == frame`` gives the front/wrist
    image names and the recorded joint state. ``only`` restricts to those
    episode directory names. Returns a list of (name, {camera: PIL.Image}, state)."""
    scenes = []
    for meta in sorted(Path(root).expanduser().rglob("metadata.jsonl")):
        ep_dir = meta.parent
        if only and ep_dir.name not in only:
            continue
        record = None
        with open(meta) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("timestep") == frame:
                    record = r
                    break
        if record is None:
            log.warning("No timestep %d in %s; skipping", frame, meta)
            continue
        try:
            imgs = {cam: load_scene_image(str(ep_dir / fname))
                    for cam, fname in record["images"].items()}
        except FileNotFoundError as e:
            log.warning("Missing image in %s (%s); skipping", ep_dir, e)
            continue
        scenes.append((ep_dir.name, imgs, record["state"]))
        if max_scenes and len(scenes) >= max_scenes:
            break
    return scenes


# --------------------------------------------------------------------------- #
# Embedding extraction
# --------------------------------------------------------------------------- #

def encode(policy, preprocessor, image, prompt, pose, device):
    """Hidden states for one (scene, prompt). ``image`` is a PIL image or a
    {camera: PIL.Image} dict. Returns (hidden [1,seq,D] cpu, pad_masks [1,seq]
    cpu, num_img_tokens)."""
    if preprocessor is not None:
        if isinstance(image, dict):
            inputs = prepare_inputs_stateful_multi(
                policy, preprocessor, image, prompt, pose, device)
        else:
            inputs = prepare_inputs_stateful(
                policy, preprocessor, image, prompt, pose, device)
    else:
        if isinstance(image, dict):
            image = image.get("front") or next(iter(image.values()))
        inputs = prepare_inputs(policy, image, prompt, device)

    hidden, _prefix_embs, pad_masks, n_img = extract_vlm_hidden_states(policy, *inputs)
    return hidden.float().cpu(), pad_masks.cpu(), n_img


def pooled_regions(hidden, pad_masks, n_img, pooling):
    """{region: [D] vector} for the full prefix / image / language slices."""
    return {
        "full_prefix": pool_embeddings(hidden, pad_masks, method=pooling).squeeze(0),
        "image_tokens": pool_region(hidden, pad_masks, 0, n_img,
                                    method=pooling).squeeze(0),
        "language_tokens": pool_region(hidden, pad_masks, n_img, hidden.shape[1],
                                       method=pooling).squeeze(0),
    }


def cosine_matrix(vecs):
    """[N, D] -> [N, N] cosine similarity (numpy)."""
    v = F.normalize(vecs, dim=-1)
    return (v @ v.T).numpy()


def per_token_cosine(hidden_a, hidden_b):
    """Per-position cosine between two [1, seq, D] sequences -> [seq] numpy."""
    a = F.normalize(hidden_a.squeeze(0), dim=-1)
    b = F.normalize(hidden_b.squeeze(0), dim=-1)
    return (a * b).sum(dim=-1).numpy()


# --------------------------------------------------------------------------- #
# Metrics (all within one scene)
# --------------------------------------------------------------------------- #

def preference_metrics(sim, labels):
    """Per-preference metrics from one scene's [N, N] similarity matrix.

    ``labels`` is the row order (base keys first, then preference keys).
    """
    idx = {k: i for i, k in enumerate(labels)}
    base_idx = [idx[b] for b in BASE_PROMPTS]
    inter_base_mean = float(np.mean(
        [sim[i, j] for i in base_idx for j in base_idx if i < j]))

    out = OrderedDict()
    for key, spec in PREFERENCE_PROMPTS.items():
        row = sim[idx[key]]
        to_base = {b: float(row[idx[b]]) for b in BASE_PROMPTS}
        nearest = max(to_base, key=to_base.get)
        adm = spec["admissible"]
        inadm = [o for o in OBJECTS if o not in adm]
        m = dict(to_base=to_base, nearest=nearest,
                 nearest_admissible=nearest in adm,
                 collapse_margin=to_base[nearest] - inter_base_mean,
                 violation_pull=None)
        if inadm:
            m["violation_pull"] = float(np.mean([to_base[o] for o in inadm])
                                        - np.mean([to_base[o] for o in adm]))
        out[key] = m
    return out, inter_base_mean


def scene_agreement(per_scene, region):
    """Cross-scene COUNTS only (no similarity averaged across scenes): per
    preference, how many scenes lean inadmissible and which base is nearest."""
    agree = OrderedDict()
    n = len(per_scene)
    for key, spec in PREFERENCE_PROMPTS.items():
        rows = [s[region]["prefs"][key] for s in per_scene]
        nearest_counts = OrderedDict((o, 0) for o in OBJECTS)
        for r in rows:
            nearest_counts[r["nearest"]] += 1
        a = dict(nearest_counts=nearest_counts,
                 n_nearest_admissible=sum(r["nearest_admissible"] for r in rows),
                 n_scenes=n)
        if rows[0]["violation_pull"] is not None:
            a["n_violating"] = sum(r["violation_pull"] > 0 for r in rows)
        agree[key] = a
    return agree


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "axes.edgecolor": GRID_COLOR, "axes.labelcolor": TEXT_SECONDARY,
        "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
        "text.color": TEXT_PRIMARY, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 9,
    })
    return plt


def _category_bounds(keys):
    """[(category, first_idx, last_idx)] for consecutive runs in ``keys``."""
    bounds = []
    for i, k in enumerate(keys):
        c = PREFERENCE_PROMPTS[k]["category"]
        if bounds and bounds[-1][0] == c:
            bounds[-1][2] = i
        else:
            bounds.append([c, i, i])
    return bounds


def plot_heatmap(sim, labels, title, save_path):
    """One scene's cosine similarity, all prompts. Single-hue sequential
    colormap over the observed off-diagonal range so small gaps stay visible."""
    plt = _plt()
    n = len(labels)
    off = sim[~np.eye(n, dtype=bool)]
    vmin, vmax = float(off.min()), 1.0
    fig, ax = plt.subplots(figsize=(0.5 * n + 3, 0.45 * n + 2))
    im = ax.imshow(sim, cmap="Blues", vmin=vmin, vmax=vmax)
    cb = plt.colorbar(im, ax=ax, fraction=0.04)
    cb.set_label("cosine similarity")
    cb.outline.set_visible(False)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7.5)
    ax.set_yticklabels(labels, fontsize=7.5)
    seps = [len(BASE_PROMPTS) - 0.5] + [
        len(BASE_PROMPTS) + e + 0.5 for _, _, e in _category_bounds(labels[len(BASE_PROMPTS):])[:-1]]
    for s in seps:
        ax.axhline(s, color="white", lw=2)
        ax.axvline(s, color="white", lw=2)
    mid = vmin + 0.6 * (vmax - vmin)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim[i, j]:.3f}", ha="center", va="center", fontsize=5,
                    color="white" if sim[i, j] > mid else TEXT_PRIMARY)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_bars(scene_region, title, save_path):
    """One scene: per preference prompt, similarity to each base prompt.
    Inadmissible objects are hatched and marked ✗; the scene's inter-base mean
    is a dashed reference (above it = inside the base-instruction cluster)."""
    plt = _plt()
    prefs = list(PREFERENCE_PROMPTS)
    w = 0.26
    x = np.arange(len(prefs))
    fig, ax = plt.subplots(figsize=(max(10, 0.75 * len(prefs)), 5))

    lo = 1.0
    for k, b in enumerate(OBJECTS):
        vals = np.array([scene_region["prefs"][p]["to_base"][b] for p in prefs])
        lo = min(lo, float(vals.min()))
        inadm = [b not in PREFERENCE_PROMPTS[p]["admissible"] for p in prefs]
        xs = x + (k - 1) * w
        bars = ax.bar(xs, vals, width=w - 0.03, color=BASE_COLORS[b],
                      linewidth=0, label=f"'{BASE_PROMPTS[b]}'")
        for bar, ex, xi, v in zip(bars, inadm, xs, vals):
            if ex:
                bar.set_hatch("////")
                bar.set_edgecolor("white")
                ax.text(xi, v, "✗", ha="center", va="bottom",
                        color=TEXT_PRIMARY, fontsize=8)

    inter = scene_region["inter_base_mean"]
    ax.axhline(inter, color=TEXT_SECONDARY, ls="--", lw=1)
    ax.text(len(prefs) - 0.5, inter, " inter-base mean", va="bottom", ha="right",
            color=TEXT_SECONDARY, fontsize=8)
    lo = min(lo, inter)
    top = 1.0
    ax.set_ylim(lo - 0.1 * (top - lo), top)
    for c, s, e in _category_bounds(prefs):
        if s > 0:
            ax.axvline(s - 0.5, color=GRID_COLOR, lw=1.5)
        ax.text((s + e) / 2, top, c, ha="center", va="bottom",
                color=TEXT_SECONDARY, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(prefs, rotation=35, ha="right")
    ax.set_ylabel("cosine similarity to base instruction")
    ax.yaxis.grid(True, color=GRID_COLOR, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False,
              title="base instruction  (hatched ✗ = inadmissible)", title_fontsize=8)
    ax.set_title(title, loc="left", fontsize=11, pad=18)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_consistency(per_scene, region, save_path):
    """Scene x preference grid of violation pull (within-scene values, never
    averaged). Orange = leans toward an inadmissible object, blue = toward an
    admissible one; blank = A_p covers every object (undefined)."""
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    prefs = list(PREFERENCE_PROMPTS)
    scenes = [s["scene"] for s in per_scene]
    M = np.full((len(prefs), len(scenes)), np.nan)
    for j, s in enumerate(per_scene):
        for i, p in enumerate(prefs):
            v = s[region]["prefs"][p]["violation_pull"]
            if v is not None:
                M[i, j] = v
    lim = float(np.nanmax(np.abs(M))) or 1e-6
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    cmap.set_bad("white")

    fig, ax = plt.subplots(figsize=(0.55 * len(scenes) + 4.5, 0.36 * len(prefs) + 1.8))
    im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    cb = plt.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("violation pull  (> 0 leans inadmissible)")
    cb.outline.set_visible(False)
    for i in range(len(prefs)):
        for j in range(len(scenes)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{1000 * M[i, j]:+.0f}", ha="center", va="center",
                        fontsize=6, color=TEXT_PRIMARY)
    n_viol = [int(np.sum(M[i] > 0)) for i in range(len(prefs))]
    ax.set_yticks(range(len(prefs)))
    ax.set_yticklabels([f"{p}  ({v}/{len(scenes)})" if not np.all(np.isnan(M[i]))
                        else f"{p}  (n/a)" for i, (p, v) in enumerate(zip(prefs, n_viol))],
                       fontsize=8)
    ax.set_xticks(range(len(scenes)))
    ax.set_xticklabels(scenes, rotation=45, ha="right", fontsize=8)
    for c, s, e in _category_bounds(prefs):
        if s > 0:
            ax.axhline(s - 0.5, color="white", lw=3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{region}: violation pull per scene (×1000; label = scenes leaning "
                 "inadmissible)", loc="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved: %s", save_path)


# --------------------------------------------------------------------------- #
# Latent-space structure: t-SNE + failure clustering
# --------------------------------------------------------------------------- #

CATEGORY_MARKERS = {"omission": "v", "sequencing": "s", "conditional": "D",
                    "quantity": "P"}


def latent_points(pooled_region, per_scene, region):
    """Scene-centred, L2-normalised points, one per (prompt, scene), plus
    per-point labels.

    Each scene's mean over all prompts is subtracted so scene-to-scene
    variation (frame, pose) doesn't dominate; normalising makes squared
    Euclidean distance = 2 - 2 cos. ``outcome`` is the *predicted* failure:
    "fail" when the preference prompt's nearest base instruction (within its
    scene) is an inadmissible object, "ok" when admissible, "n/a" when every
    object is admissible, "base" for the base instructions themselves.
    """
    keys = list(pooled_region)
    stacked = torch.stack([torch.stack(pooled_region[k]) for k in keys])  # [P, S, D]
    stacked = stacked - stacked.mean(dim=0, keepdim=True)
    X = F.normalize(stacked, dim=-1).reshape(-1, stacked.shape[-1]).numpy()
    n_scenes = stacked.shape[1]
    meta = []
    for k in keys:
        for si in range(n_scenes):
            if k in BASE_PROMPTS:
                outcome = "base"
            else:
                m = per_scene[si][region]["prefs"][k]
                if m["violation_pull"] is None:
                    outcome = "n/a"
                else:
                    outcome = "ok" if m["nearest_admissible"] else "fail"
            nearest = k if k in BASE_PROMPTS else per_scene[si][region]["prefs"][k]["nearest"]
            meta.append(dict(prompt=k, scene=per_scene[si]["scene"], outcome=outcome,
                             nearest=nearest,
                             category="base" if k in BASE_PROMPTS
                             else PREFERENCE_PROMPTS[k]["category"]))
    return X, meta


def tsne(X, perplexity=30.0, n_iter=1000, seed=0):
    """Exact t-SNE (van der Maaten & Hinton 2008) in numpy -- a few hundred
    points, so O(N^2) is fine and avoids a scikit-learn dependency. Uses
    early exaggeration (12x, 250 iters), momentum 0.5 -> 0.8, gains, and a
    small-scale PCA initialisation for run-to-run stability."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    perplexity = min(perplexity, (n - 1) / 3)
    sq = np.sum(X ** 2, axis=1)
    D = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0)

    # Conditional P_{j|i} by binary search on the Gaussian precision.
    P = np.zeros((n, n))
    target = np.log(perplexity)
    for i in range(n):
        d = np.delete(D[i], i)
        lo, hi, beta = 0.0, np.inf, 1.0
        for _ in range(100):
            p = np.exp(-(d - d.min()) * beta)
            sp = p.sum()
            H = np.log(sp) + beta * np.sum((d - d.min()) * p) / sp
            if abs(H - target) < 1e-5:
                break
            if H > target:
                lo = beta
                beta = beta * 2 if hi == np.inf else (beta + hi) / 2
            else:
                hi = beta
                beta = (beta + lo) / 2
        P[i, np.arange(n) != i] = p / sp
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)

    Xc = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    Y = Xc @ Vt[:2].T
    Y = Y / Y[:, 0].std() * 1e-4
    rng = np.random.default_rng(seed)
    Y += rng.normal(scale=1e-6, size=Y.shape)

    lr = max(n / 12 / 4, 50.0)
    update = np.zeros_like(Y)
    gains = np.ones_like(Y)
    for it in range(n_iter):
        exag = 12.0 if it < 250 else 1.0
        mom = 0.5 if it < 250 else 0.8
        sqy = np.sum(Y ** 2, axis=1)
        num = 1.0 / (1.0 + sqy[:, None] + sqy[None, :] - 2 * Y @ Y.T)
        np.fill_diagonal(num, 0)
        Q = np.maximum(num / num.sum(), 1e-12)
        PQ = (exag * P - Q) * num
        grad = 4 * ((np.diag(PQ.sum(axis=1)) - PQ) @ Y)
        gains = np.where(np.sign(grad) != np.sign(update), gains + 0.2, gains * 0.8)
        gains = np.maximum(gains, 0.01)
        update = mom * update - lr * gains * grad
        Y = Y + update
        Y = Y - Y.mean(axis=0)
    kl = float(np.sum(P * np.log(P / Q)))
    return Y, kl


def knn_failure_purity(X, meta, k=10, n_perm=2000, seed=0):
    """Do predicted failures sit next to other failures in the latent space?

    Over preference points with a defined outcome (fail/ok), for each FAIL
    point take its k nearest labelled neighbours (cosine, full-dimensional
    space -- not the 2-D t-SNE map) and measure the fraction that also fail.
    ``cross_prompt`` repeats this excluding neighbours from the same prompt,
    i.e. asks whether failures of *different* preference instructions group
    together, not just the same prompt across scenes. ``cross_category``
    also excludes neighbours from the same preference category, so shared
    phrasing templates (every omission prompt starts "put everything on the
    tray except ...") can't produce the clustering on their own. Chance = the
    fail base rate; p-value from a label permutation test.
    """
    idx = [i for i, m in enumerate(meta) if m["outcome"] in ("fail", "ok")]
    Z = X[idx]
    fail = np.array([meta[i]["outcome"] == "fail" for i in idx])
    prompts = np.array([meta[i]["prompt"] for i in idx])
    cats = np.array([meta[i]["category"] for i in idx])
    S = Z @ Z.T
    np.fill_diagonal(S, -np.inf)
    same = prompts[:, None] == prompts[None, :]
    S_cross = np.where(same, -np.inf, S)
    S_cat = np.where(cats[:, None] == cats[None, :], -np.inf, S)

    def purity(labels, sim):
        nn = np.argsort(-sim, axis=1)[:, :k]
        frac = labels[nn].mean(axis=1)
        return float(frac[labels].mean()) if labels.any() else float("nan")

    rng = np.random.default_rng(seed)
    out = {"k": k, "n_points": int(len(idx)), "n_fail": int(fail.sum()),
           "fail_rate": float(fail.mean())}
    for name, sim in (("all", S), ("cross_prompt", S_cross), ("cross_category", S_cat)):
        obs = purity(fail, sim)
        null = np.array([purity(rng.permutation(fail), sim) for _ in range(n_perm)])
        out[name] = {"purity": obs, "null_mean": float(null.mean()),
                     "p_value": float((1 + np.sum(null >= obs)) / (1 + n_perm))}
    return out


def refers_to(spec):
    """Objects a preference instruction is a version of: the objects it names
    (omission: excluded objects; sequencing: objects named in the sentence) or,
    for conditional / quantity, the objects its description picks out.
    "one object" / "two objects" refer to no particular object -> []."""
    if spec["category"] == "omission":
        return list(spec["excluded"])
    if spec["category"] == "sequencing":
        # Objects named in the sentence -- "put the pouch on the tray last"
        # names only the pouch, though its order also constrains the others.
        words = spec["prompt"].replace(",", " ").split()
        return [o for o in OBJECTS if o in words]
    if spec["category"] == "quantity" and len(spec["cls"]) == len(OBJECTS):
        return []
    return list(spec["admissible"])


# Base instruction -> the preference instructions that are versions of it.
VERSION_GROUPS = OrderedDict(
    (o, [k for k, spec in PREFERENCE_PROMPTS.items() if o in refers_to(spec)])
    for o in OBJECTS)


def _style_map_axes(ax, pts):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    px, py = 0.08 * (x1 - x0 or 1), 0.08 * (y1 - y0 or 1)
    ax.set_xlim(x0 - px, x1 + px)
    ax.set_ylim(y0 - py, y1 + py)


def plot_tsne(Y, meta, kl, title, save_path):
    """All-scene t-SNE, one panel per base instruction: that instruction's
    preference versions (one point per scene) plus all three base
    instructions as anchors, the panel's own base ringed. Preference points
    are filled with the colour of their nearest base instruction in that
    scene (full 2048-d); marker shape = preference category. Positions come
    from one t-SNE of every prompt, so the panels share coordinates."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(OBJECTS), figsize=(7 * len(OBJECTS), 6.8))
    prompt_arr = np.array([m["prompt"] for m in meta])
    cat_arr = np.array([m["category"] for m in meta])
    near_arr = np.array([m["nearest"] for m in meta])
    for ax, obj in zip(axes, OBJECTS):
        in_group = np.isin(prompt_arr, VERSION_GROUPS[obj])
        for cat, marker in CATEGORY_MARKERS.items():
            for b in BASE_PROMPTS:
                mk = in_group & (cat_arr == cat) & (near_arr == b)
                if mk.any():
                    ax.scatter(Y[mk, 0], Y[mk, 1], s=34, marker=marker,
                               color=BASE_COLORS[b], edgecolor="white", linewidth=0.5,
                               alpha=0.85, zorder=2)
        for b in BASE_PROMPTS:
            mk = prompt_arr == b
            ax.scatter(Y[mk, 0], Y[mk, 1], s=90, color=BASE_COLORS[b],
                       edgecolor=TEXT_PRIMARY if b == obj else "white",
                       linewidth=1.8 if b == obj else 1.2, zorder=3)
        _style_map_axes(ax, Y[in_group | np.isin(prompt_arr, list(BASE_PROMPTS))])
        ax.set_title(f"'{BASE_PROMPTS[obj]}' + {len(VERSION_GROUPS[obj])} preference "
                     "versions", loc="left", fontsize=10)
    fig.legend(handles=_scene_tsne_legend(), loc="lower center", ncol=4,
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"{title}   (KL {kl:.3f})", x=0.01, ha="left", fontsize=11)
    plt.tight_layout(rect=(0, 0.05, 1, 0.97))
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Saved: %s", save_path)


def scene_tsne(pooled_scene, perplexity=5.0, seed=0):
    """t-SNE of one scene's prompts (one point per prompt). Centred on the
    scene's mean over prompts and L2-normalised, like the global map."""
    keys = list(pooled_scene)
    V = torch.stack([pooled_scene[k] for k in keys])
    V = F.normalize(V - V.mean(dim=0, keepdim=True), dim=-1).numpy()
    Y, kl = tsne(V, perplexity=perplexity, seed=seed)
    return keys, Y, kl


def draw_scene_tsne(ax, keys, Y, scene_region, group, compact=False):
    """One scene's map on ``ax`` for one base instruction ``group``: its
    preference versions plus all three base instructions (the group's own
    base ringed). Colour always means an object:
      - base instructions: large dots in their object colour;
      - preferences: marker shape = category, fill = the object colour of the
        base instruction nearest in the full 2048-d space (the known
        behaviour the VLA latches onto), thin line to that base.
    Positions come from the scene's t-SNE over every prompt, so the three
    group panels of a scene share coordinates. No text labels; per-prompt
    coordinates are saved in results.json under per_scene[..][region]["tsne"].
    """
    pos = {k: Y[i] for i, k in enumerate(keys)}
    versions = VERSION_GROUPS[group]
    for k in versions:
        nb = scene_region["prefs"][k]["nearest"]
        ax.plot([pos[k][0], pos[nb][0]], [pos[k][1], pos[nb][1]],
                color=BASE_COLORS[nb], lw=0.8, alpha=0.5, zorder=1)
    for k in versions:
        colour = BASE_COLORS[scene_region["prefs"][k]["nearest"]]
        ax.scatter(*pos[k], s=(45 if compact else 90),
                   marker=CATEGORY_MARKERS[PREFERENCE_PROMPTS[k]["category"]],
                   color=colour, edgecolor="white", linewidth=0.8, zorder=3)
    for b in BASE_PROMPTS:
        ax.scatter(*pos[b], s=(110 if compact else 220), color=BASE_COLORS[b],
                   edgecolor=TEXT_PRIMARY if b == group else "white",
                   linewidth=(1.8 if b == group else 1.5), zorder=5)
    _style_map_axes(ax, np.array([pos[k] for k in versions + list(BASE_PROMPTS)]))


def _scene_tsne_legend(fig_or_ax=None, compact=False):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", markersize=9, color=BASE_COLORS[b],
                label=f"'{BASE_PROMPTS[b]}'") for b in BASE_PROMPTS]
    h += [Line2D([], [], marker="o", ls="", markersize=9, markerfacecolor=PREF_COLOR,
                 markeredgecolor=TEXT_PRIMARY, markeredgewidth=1.8,
                 label="ringed = the base instruction the panel's preferences are versions of")]
    h += [Line2D([], [], marker=mk, ls="", markersize=7, color=PREF_COLOR, label=c)
          for c, mk in CATEGORY_MARKERS.items()]
    h += [Line2D([], [], color=PREF_COLOR, lw=1,
                 label="preference fill (+ line) = nearest base instruction (full 2048-d)")]
    return h


def plot_scene_tsne(keys, Y, kl, scene_region, title, save_path, perplexity):
    """One scene, one panel per base instruction (shared t-SNE coordinates)."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(OBJECTS), figsize=(7 * len(OBJECTS), 6.2))
    for ax, obj in zip(axes, OBJECTS):
        draw_scene_tsne(ax, keys, Y, scene_region, obj)
        ax.set_title(f"'{BASE_PROMPTS[obj]}' + {len(VERSION_GROUPS[obj])} preference "
                     "versions", loc="left", fontsize=10)
    fig.legend(handles=_scene_tsne_legend(), loc="lower center", ncol=4,
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"{title}   (t-SNE perplexity {perplexity:g}, KL {kl:.3f})",
                 x=0.01, ha="left", fontsize=11)
    plt.tight_layout(rect=(0, 0.06, 1, 0.96))
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_scene_tsne_grid(panels, region, group, save_path, perplexity):
    """Small multiples for one base instruction: one per-scene map per panel."""
    plt = _plt()
    n = len(panels)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.4 * rows), squeeze=False)
    for ax, (name, keys, Y, kl, scene_region) in zip(axes.flat, panels):
        draw_scene_tsne(ax, keys, Y, scene_region, group, compact=True)
        ax.set_title(name, loc="left", fontsize=9)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.legend(handles=_scene_tsne_legend(compact=True), loc="lower center",
               ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{region}: '{BASE_PROMPTS[group]}' and its preference versions, "
                 f"per scene (t-SNE perplexity {perplexity:g}; colour = nearest base "
                 "instruction)", x=0.01, ha="left", fontsize=12)
    plt.tight_layout(rect=(0, 0.04, 1, 0.98))
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("  Saved: %s", save_path)


def plot_per_token(sims_by_base, n_img, n_valid, title, save_path):
    """Per-position cosine between one preference prompt and each base prompt
    on a single scene (language region zoomed; image region summarised)."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 4))
    for b, (s, inadm) in sims_by_base.items():
        lang = s[n_img:n_valid]
        ax.plot(np.arange(n_img, n_valid), lang, color=BASE_COLORS[b], lw=2,
                ls="--" if inadm else "-",
                label=f"vs '{BASE_PROMPTS[b]}'" + ("  (inadmissible)" if inadm else "")
                      + f"   img mean {s[:n_img].mean():.3f}")
    ax.set_xlabel("token position (language region; image tokens 0.."
                  f"{n_img - 1} summarised in legend). Positions are aligned by "
                  "index, so tokens after the prompts diverge in length are offset")
    ax.set_ylabel("cosine similarity")
    ax.grid(True, color=GRID_COLOR, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=8)
    ax.set_title(title, loc="left", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #

def write_summary(per_scene, args, path, clustering=None):
    n = len(per_scene)
    lines = [
        "# VLA baseline: preference vs base instruction embeddings",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- scenes: {n}  (pooling: {args.pooling}, state-conditioned: {not args.no_state})",
        "- every prompt is encoded on every scene; all similarities are within-scene",
        "- admissible set A_p: omission G_b\\E_p · sequencing <_p-minimal · "
        "conditional {x: phi_p} · quantity class c",
        "- violation pull = mean sim(pref, X\\A_p) − mean sim(pref, A_p); "
        "**> 0 leans toward an inadmissible object** (n/a when A_p = X)",
        "",
        "## Scene agreement",
        "",
        "Counts of scenes, per region: `viol` = scenes with violation pull > 0, "
        "`adm` = scenes whose nearest base instruction is admissible.",
        "",
        "| category | preference | A_p | " + " | ".join(
            f"{r} viol / adm" for r in REGIONS) + " | nearest (language) |",
        "|---|---|---|" + "---|" * (len(REGIONS) + 1),
    ]
    agrees = {r: scene_agreement(per_scene, r) for r in REGIONS}
    for key, spec in PREFERENCE_PROMPTS.items():
        cells = []
        for r in REGIONS:
            a = agrees[r][key]
            v = f"{a['n_violating']}/{n}" if "n_violating" in a else "n/a"
            cells.append(f"{v} / {a['n_nearest_admissible']}/{n}")
        nc = agrees["language_tokens"][key]["nearest_counts"]
        near = ", ".join(f"{o}:{c}" for o, c in nc.items() if c)
        lines.append(f"| {spec['category']} | {spec['prompt']} | "
                     f"{', '.join(spec['admissible'])} | " + " | ".join(cells)
                     + f" | {near} |")

    if clustering:
        lines += ["", "## Failure clustering in latent space", "",
                  "Predicted failure = nearest base instruction (within scene) is "
                  "inadmissible. Scene-centred, L2-normalised 2048-d pooled states; "
                  "kNN fail purity = mean fraction of a failure's k nearest labelled "
                  "neighbours that also fail. *cross-prompt* ignores neighbours from "
                  "the same prompt. p from a 2000-shuffle label permutation.", "",
                  "| region | failures | chance | purity (all) | p | purity (cross-prompt) | p "
                  "| purity (cross-category) | p |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r, c in clustering.items():
            lines.append(f"| {r} | {c['n_fail']}/{c['n_points']} | {c['fail_rate']:.3f} | "
                         f"{c['all']['purity']:.3f} | {c['all']['p_value']:.4f} | "
                         f"{c['cross_prompt']['purity']:.3f} | {c['cross_prompt']['p_value']:.4f} | "
                         f"{c['cross_category']['purity']:.3f} | "
                         f"{c['cross_category']['p_value']:.4f} |")

    for s in per_scene:
        lines += ["", f"## Scene {s['scene']}", ""]
        for r in REGIONS:
            sr = s[r]
            lines += [f"### {r}  (inter-base {sr['inter_base_mean']:.4f})", "",
                      "| preference | " + " | ".join(f"sim→{o}" for o in OBJECTS)
                      + " | nearest | violation pull |",
                      "|---|" + "---|" * (len(OBJECTS) + 2)]
            for key, spec in PREFERENCE_PROMPTS.items():
                p = sr["prefs"][key]
                sims = [f"{p['to_base'][o]:.4f}" + ("" if o in spec["admissible"] else " ✗")
                        for o in OBJECTS]
                vp = "n/a" if p["violation_pull"] is None else f"{p['violation_pull']:+.4f}"
                lines.append(f"| {key} | " + " | ".join(sims)
                             + f" | {p['nearest']} | {vp} |")
            lines.append("")
    Path(path).write_text("\n".join(lines))
    log.info("  Saved: %s", path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="VLA baseline: preference vs base instruction embeddings (PI05)")
    parser.add_argument("--checkpoint", default="ant0nh/pi05_500_30k",
                        help="Path or HF repo of the checkpoint to analyse")
    parser.add_argument("--episode_root", default="vla_inputs",
                        help="Root searched recursively for episode dirs "
                             "(metadata.jsonl + front/wrist frames)")
    parser.add_argument("--scenes", nargs="*", default=None,
                        help="Only these episode directory names (e.g. 001_banana)")
    parser.add_argument("--frame", type=int, default=0,
                        help="Episode timestep to use as the scene (default 0)")
    parser.add_argument("--max_scenes", type=int, default=None)
    parser.add_argument("--image_path", default=None,
                        help="Single scene image instead of episodes (uses INITIAL_POSE)")
    parser.add_argument("--live", action="store_true",
                        help="Capture one scene from the RealSense instead of episodes")
    parser.add_argument("--output", default="preference_baseline",
                        help="Output directory (a timestamped run dir is created inside)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--pooling", default="mean", choices=["mean", "last", "cls"])
    parser.add_argument("--no_state", action="store_true",
                        help="Bare stateless prompts (front camera only) instead of "
                             "the checkpoint preprocessor")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--perplexity", type=float, default=30.0,
                        help="Global (all-scene) t-SNE perplexity (default 30)")
    parser.add_argument("--scene_perplexity", type=float, default=5.0,
                        help="Per-scene t-SNE perplexity; each scene has only "
                             "24 points (default 5)")
    parser.add_argument("--knn_k", type=int, default=10,
                        help="Neighbours for the failure-clustering purity test")
    args = parser.parse_args()

    # --- Scenes ---
    if args.image_path:
        scenes = [(Path(args.image_path).stem, load_scene_image(args.image_path),
                   INITIAL_POSE)]
    elif args.live:
        from vlm import RealSenseCamera, capture_scene
        cam = RealSenseCamera()
        cam.start()
        try:
            scenes = [("live", capture_scene(cam), INITIAL_POSE)]
        finally:
            cam.stop()
    else:
        scenes = load_episode_scenes(args.episode_root, args.frame, args.max_scenes,
                                     set(args.scenes) if args.scenes else None)
    if not scenes:
        log.error("No scenes found.")
        return
    log.info("%d scene(s): %s", len(scenes), [s[0] for s in scenes])

    # --- Model ---
    log.info("Loading policy from %s (dtype=%s) ...", args.checkpoint, args.dtype)
    policy = load_policy(args.checkpoint, args.device, args.dtype)
    log.info("Checkpoint image_features: %s", list(policy.config.image_features))
    preprocessor = None
    if not args.no_state:
        preprocessor = build_preprocessor(args.checkpoint, policy.config)

    labels = list(BASE_PROMPTS) + list(PREFERENCE_PROMPTS)
    prompt_text = {**BASE_PROMPTS,
                   **{k: v["prompt"] for k, v in PREFERENCE_PROMPTS.items()}}

    run_dir = Path(args.output) / time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Every prompt on every scene; metrics within the scene ---
    per_scene = []
    pooled_all = {r: {k: [] for k in labels} for r in REGIONS}
    scene_panels = {r: [] for r in REGIONS}

    for si, (name, image, pose) in enumerate(scenes, 1):
        log.info("=== Scene [%d/%d]: %s ===", si, len(scenes), name)
        t0 = time.time()
        hidden, pads, pooled = {}, {}, {r: {} for r in REGIONS}
        n_img = None
        for k in labels:
            h, pm, n_img = encode(policy, preprocessor, image, prompt_text[k], pose,
                                  args.device)
            hidden[k], pads[k] = h, pm
            for r, v in pooled_regions(h, pm, n_img, args.pooling).items():
                pooled[r][k] = v
                pooled_all[r][k].append(v)

        scene_res = {"scene": name}
        scene_sims = {}
        for r in REGIONS:
            sim = cosine_matrix(torch.stack([pooled[r][k] for k in labels]))
            scene_sims[r] = sim
            prefs, inter = preference_metrics(sim, labels)
            scene_res[r] = {"inter_base_mean": inter, "prefs": prefs,
                            "similarity": sim.tolist()}
        per_scene.append(scene_res)

        lang = scene_res["language_tokens"]["prefs"]
        n_def = sum(m["violation_pull"] is not None for m in lang.values())
        n_viol = sum((m["violation_pull"] or 0) > 0 for m in lang.values())
        log.info("  language_tokens: %d/%d preferences lean inadmissible  (%.1fs)",
                 n_viol, n_def, time.time() - t0)

        if not args.no_plots:
            sdir = run_dir / "scenes" / name
            sdir.mkdir(parents=True, exist_ok=True)
            for r in REGIONS:
                plot_heatmap(scene_sims[r], labels, f"{r}: cosine similarity — scene {name}",
                             sdir / f"heatmap_{r}.png")
                plot_bars(scene_res[r], f"{r}: preference → base instruction — scene {name}",
                          sdir / f"bars_{r}.png")
                keys, Y, kl = scene_tsne(pooled[r], perplexity=args.scene_perplexity)
                scene_res[r]["tsne"] = {"kl": kl, "points": {
                    k: [float(y[0]), float(y[1])] for k, y in zip(keys, Y)}}
                scene_panels[r].append((name, keys, Y, kl, scene_res[r]))
                plot_scene_tsne(keys, Y, kl, scene_res[r],
                                f"{r}: base instructions and their preference versions "
                                f"— scene {name}",
                                sdir / f"tsne_{r}.png", args.scene_perplexity)
            if si == 1:
                for key, spec in PREFERENCE_PROMPTS.items():
                    n_valid = int(max(pads[key].sum(),
                                      *(pads[b].sum() for b in BASE_PROMPTS)))
                    sims = {b: (per_token_cosine(hidden[key], hidden[b]),
                                b not in spec["admissible"]) for b in BASE_PROMPTS}
                    plot_per_token(sims, n_img, n_valid,
                                   f"'{spec['prompt']}' vs base instructions — scene {name}",
                                   sdir / f"per_token_{key.replace(' ', '_').replace('>', '-')}.png")
            log.info("  Saved per-scene plots: %s/", sdir)

    # --- Console summary: scene agreement per region ---
    n = len(per_scene)
    for r in REGIONS:
        agree = scene_agreement(per_scene, r)
        print(f"\n--- {r}: scenes leaning inadmissible / nearest admissible (of {n}) ---")
        cat = None
        for key, spec in PREFERENCE_PROMPTS.items():
            if spec["category"] != cat:
                cat = spec["category"]
                print(f"  [{cat}]")
            a = agree[key]
            viol = f"{a['n_violating']:2d}/{n}" if "n_violating" in a else "  n/a"
            near = " ".join(f"{o}:{c}" for o, c in a["nearest_counts"].items())
            print(f"    {key:22s} A={','.join(spec['admissible']):18s} "
                  f"viol {viol}   adm {a['n_nearest_admissible']:2d}/{n}   nearest {near}")

    # --- Latent-space structure (scene-centred; computed before saving) ---
    latent = {}
    if n > 1:
        for r in REGIONS:
            X, meta = latent_points(pooled_all[r], per_scene, r)
            purity = knn_failure_purity(X, meta, k=args.knn_k)
            latent[r] = {"X": X, "meta": meta, "purity": purity}
            a, c = purity["all"], purity["cross_prompt"]
            print(f"\n--- {r}: failure clustering (kNN k={purity['k']}, "
                  f"{purity['n_fail']}/{purity['n_points']} predicted failures, "
                  f"chance {purity['fail_rate']:.3f}) ---")
            print(f"    fail purity (all neighbours):   {a['purity']:.3f}  "
                  f"(null {a['null_mean']:.3f}, p={a['p_value']:.4f})")
            print(f"    fail purity (other prompts):    {c['purity']:.3f}  "
                  f"(null {c['null_mean']:.3f}, p={c['p_value']:.4f})")
            g = purity["cross_category"]
            print(f"    fail purity (other categories): {g['purity']:.3f}  "
                  f"(null {g['null_mean']:.3f}, p={g['p_value']:.4f})")

    # --- Save results ---
    results = {
        "checkpoint": args.checkpoint,
        "pooling": args.pooling,
        "state_conditioned": preprocessor is not None,
        "objects": OBJECTS,
        "base_prompts": BASE_PROMPTS,
        "preference_prompts": PREFERENCE_PROMPTS,
        "labels": labels,
        "scenes": [s[0] for s in scenes],
        "scene_agreement": {r: scene_agreement(per_scene, r) for r in REGIONS},
        "per_scene": per_scene,
        "failure_clustering": {r: v["purity"] for r, v in latent.items()},
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("  Saved: %s", run_dir / "results.json")
    write_summary(per_scene, args, run_dir / "summary.md",
                  {r: v["purity"] for r, v in latent.items()})

    # --- Cross-scene views (counts / scene-centred, never scene-averaged sims) ---
    if not args.no_plots:
        for r in REGIONS:
            plot_consistency(per_scene, r, run_dir / f"consistency_{r}.png")
            if scene_panels[r]:
                for obj in OBJECTS:
                    plot_scene_tsne_grid(scene_panels[r], r, obj,
                                         run_dir / f"tsne_scenes_{r}_{obj}.png",
                                         args.scene_perplexity)
            if r in latent:
                log.info("  t-SNE (%s) ...", r)
                Y, kl = tsne(latent[r]["X"], perplexity=args.perplexity)
                with open(run_dir / f"tsne_{r}.json", "w") as f:
                    json.dump({"kl": kl, "perplexity": args.perplexity,
                               "points": [dict(m, x=float(y[0]), y=float(y[1]))
                                          for m, y in zip(latent[r]["meta"], Y)]},
                              f, indent=1)
                plot_tsne(Y, latent[r]["meta"], kl,
                          f"{r}: t-SNE of pooled VLM hidden states, scene-centred "
                          f"({n} scenes, perplexity {args.perplexity:g})",
                          run_dir / f"tsne_{r}.png")

    log.info("Done. Results in %s/", run_dir)


if __name__ == "__main__":
    main()
