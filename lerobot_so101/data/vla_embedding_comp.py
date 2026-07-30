"""
Extract and compare VLM final hidden states from a finetuned PI05 model.

Given the same camera frame, vary the text instruction and analyse whether
the VLM backbone's final representation — the KV cache boundary that
conditions the action expert — separates preference variants or collapses them.

Architecture target:
    PI05Policy.model.paligemma_with_expert.paligemma.model.language_model
    → forward() → .last_hidden_state  [B, seq_len, 2048]

    This is the output of the Gemma 2B VLM after 18 transformer layers
    with prefix-LM attention over [256 image tokens | N language tokens].
    It gets cached as KV pairs and served to the action expert during
    all denoising steps — if two instructions produce identical states here,
    the action expert has zero information to differentiate them.

Analyses produced:
    1. Cosine similarity matrix (pooled hidden states)
    2. Per-token position analysis (image vs language token regions)
    3. UMAP projection
    4. Action output comparison (optional, requires full denoising)

Usage:
    python extract_pi05_embeddings.py \
        --checkpoint /path/to/finetuned/checkpoint \
        --image /path/to/saved_frame.png \
        --output embedding_analysis

    # Compare base vs finetuned
    python extract_pi05_embeddings.py \
        --checkpoint /path/to/finetuned/checkpoint \
        --checkpoint2 lerobot/pi05_base \
        --image /path/to/saved_frame.png \
        --output embedding_analysis
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# --------------------------------------------------------------------------- #
# 1. Model loading
# --------------------------------------------------------------------------- #

def load_policy(checkpoint_path: str, device: str = "cuda"):
    """Load a PI05Policy from a pretrained checkpoint."""
    from lerobot.policies.pi05 import PI05Policy

    policy = PI05Policy.from_pretrained(checkpoint_path)
    policy = policy.to(device).eval()
    return policy


# --------------------------------------------------------------------------- #
# 2. Prepare inputs
# --------------------------------------------------------------------------- #

def prepare_inputs(policy, image_path: str, instruction: str, device: str = "cuda"):
    """
    Convert a raw image + instruction string into the tensors that
    PI05Pytorch.embed_prefix() expects: (images, img_masks, tokens, masks).
    """
    from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch
    from transformers import AutoTokenizer

    config = policy.config

    # --- Image ---
    img = Image.open(image_path).convert("RGB")
    img_tensor = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
    img_size = config.image_resolution[0]
    img_tensor = resize_with_pad_torch(img_tensor, img_size, img_size)
    img_tensor = img_tensor.squeeze(0).permute(2, 0, 1).unsqueeze(0).to(device)

    images = [img_tensor]
    img_masks = [torch.ones(1, dtype=torch.bool, device=device)]

    # --- Language tokens ---
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    encoded = tokenizer(
        instruction,
        return_tensors="pt",
        padding="max_length",
        max_length=config.max_token_length,
        truncation=True,
    ).to(device)

    tokens = encoded["input_ids"]
    masks = encoded["attention_mask"].bool()

    return images, img_masks, tokens, masks


# --------------------------------------------------------------------------- #
# 3. Extract VLM final hidden states
# --------------------------------------------------------------------------- #

@torch.no_grad()
def extract_vlm_hidden_states(policy, images, img_masks, tokens, masks):
    """
    Run the prefix through the VLM backbone and return the final hidden
    states — the representation that gets cached as KV pairs for the
    action expert.

    Returns:
        hidden_states: [B, seq_len, 2048] — VLM final norm output
        prefix_embs:   [B, seq_len, 2048] — pre-transformer concatenated embeddings
        pad_masks:     [B, seq_len]        — valid token mask
        num_img_tokens: int                — number of image patch tokens (256)
    """
    from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks

    model = policy.model

    # Embed prefix: SigLIP image patches + language token embeddings
    prefix_embs, pad_masks, att_masks = model.embed_prefix(
        images, img_masks, tokens, masks
    )

    # Number of image tokens = prefix length - language length
    num_img_tokens = prefix_embs.shape[1] - tokens.shape[1]

    # Build attention masks and position IDs
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    att_2d_masks_4d = att_2d_masks[:, None, :, :]
    att_2d_masks_4d = torch.where(att_2d_masks_4d, 0.0, -1e9)

    # Forward through VLM backbone only
    vlm = model.paligemma_with_expert.paligemma.model.language_model
    vlm_output = vlm.forward(
        inputs_embeds=prefix_embs,
        attention_mask=att_2d_masks_4d,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=False,
    )

    hidden_states = vlm_output.last_hidden_state  # [B, seq_len, 2048]
    return hidden_states, prefix_embs, pad_masks, num_img_tokens


# --------------------------------------------------------------------------- #
# 4. Extract action outputs (full denoising)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def extract_action_outputs(policy, images, img_masks, tokens, masks):
    """
    Run full inference (prefix + denoising loop) and return the predicted
    action chunk. This tests whether differences in the VLM hidden states
    actually propagate through the action expert into different trajectories.

    Returns:
        actions: [B, chunk_size, action_dim]
    """
    model = policy.model
    actions = model.sample_actions(images, img_masks, tokens, masks)
    return actions


# --------------------------------------------------------------------------- #
# 5. Pooling utilities
# --------------------------------------------------------------------------- #

def pool_embeddings(hidden_states, pad_masks, method="mean"):
    """Pool sequence-level hidden states into a single vector."""
    if method == "mean":
        mask = pad_masks.unsqueeze(-1).float()
        pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    elif method == "last":
        lengths = pad_masks.sum(dim=1) - 1
        pooled = hidden_states[torch.arange(len(lengths)), lengths]
    elif method == "cls":
        pooled = hidden_states[:, 0, :]
    else:
        raise ValueError(f"Unknown pooling method: {method}")
    return pooled


def pool_region(hidden_states, pad_masks, start, end, method="mean"):
    """Pool a specific slice of the sequence (e.g. image-only or language-only)."""
    region_hidden = hidden_states[:, start:end, :]
    region_masks = pad_masks[:, start:end]
    return pool_embeddings(region_hidden, region_masks, method)


# --------------------------------------------------------------------------- #
# 6. Analysis functions
# --------------------------------------------------------------------------- #

def cosine_similarity_matrix(embeddings_dict):
    """Compute pairwise cosine similarity. Returns (results_dict, matrix, names)."""
    names = list(embeddings_dict.keys())
    vecs = torch.stack([embeddings_dict[n] for n in names])
    vecs = F.normalize(vecs, dim=-1)
    sim_matrix = (vecs @ vecs.T).numpy()

    results = {}
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            results[(ni, nj)] = float(sim_matrix[i, j])
    return results, sim_matrix, names


def per_token_cosine_similarity(hidden_states_a, hidden_states_b):
    """
    Compute per-position cosine similarity between two hidden state sequences.
    This reveals WHERE in the sequence the representations differ:
    - If image token positions (0..255) are identical but language positions
      differ, the VLM isn't grounding language in vision.
    - If image positions also differ, the model IS attending to different
      scene regions based on the instruction.

    Returns: [seq_len] tensor of cosine similarities per position.
    """
    a = F.normalize(hidden_states_a.squeeze(0), dim=-1)  # [seq, 2048]
    b = F.normalize(hidden_states_b.squeeze(0), dim=-1)
    return (a * b).sum(dim=-1).cpu()  # [seq]


# --------------------------------------------------------------------------- #
# 7. Plotting
# --------------------------------------------------------------------------- #

def plot_similarity_matrix(sim_matrix, names, title, save_path):
    """Heatmap of pairwise cosine similarities."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=0.8, vmax=1.0)
    plt.colorbar(im, ax=ax, label='Cosine Similarity')

    short_names = [n[:45] + '...' if len(n) > 45 else n for n in names]
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)

    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f'{sim_matrix[i,j]:.4f}',
                    ha='center', va='center', fontsize=7,
                    color='white' if sim_matrix[i,j] > 0.95 else 'black')

    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_per_token_similarity(per_token_sims, pair_labels, num_img_tokens, title, save_path):
    """
    Plot per-position cosine similarity for multiple instruction pairs.
    Vertical line separates image token region from language token region.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))

    for label, sims in zip(pair_labels, per_token_sims):
        ax.plot(sims.numpy(), label=label, alpha=0.8, linewidth=1.2)

    ax.axvline(x=num_img_tokens, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(num_img_tokens - 2, ax.get_ylim()[0] + 0.01, 'image tokens',
            ha='right', va='bottom', fontsize=9, color='red', alpha=0.7)
    ax.text(num_img_tokens + 2, ax.get_ylim()[0] + 0.01, 'language tokens',
            ha='left', va='bottom', fontsize=9, color='red', alpha=0.7)

    ax.set_xlabel('Token position')
    ax.set_ylabel('Cosine similarity')
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc='lower left')
    ax.set_ylim(0.5, 1.02)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_umap(embeddings_dict, labels, title, save_path):
    """UMAP projection colored by instruction group."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    try:
        import umap
    except ImportError:
        print("  Skipping UMAP (pip install umap-learn)")
        return

    names = list(embeddings_dict.keys())
    vecs = torch.stack([embeddings_dict[n] for n in names]).numpy()

    if len(vecs) < 3:
        print("  Need at least 3 instructions for UMAP")
        return

    n_neighbors = min(5, len(vecs) - 1)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.3, random_state=42)
    projected = reducer.fit_transform(vecs)

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = list(set(labels.values()))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    color_map = {l: colors[i] for i, l in enumerate(unique_labels)}

    for i, name in enumerate(names):
        label = labels[name]
        ax.scatter(projected[i, 0], projected[i, 1],
                   c=[color_map[label]], s=120, zorder=5, edgecolors='white', linewidth=0.5)
        ax.annotate(name[:35], (projected[i, 0], projected[i, 1]),
                    fontsize=7, ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points')

    for label, color in color_map.items():
        ax.scatter([], [], c=[color], label=label, s=80)
    ax.legend(loc='best', fontsize=9)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# --------------------------------------------------------------------------- #
# 8. Main pipeline
# --------------------------------------------------------------------------- #

def run_comparison(
    checkpoint_path: str,
    image_path: str,
    instructions: list[dict],
    output_dir: str = "embedding_analysis",
    device: str = "cuda",
    pooling: str = "mean",
    checkpoint2_path: str | None = None,
    compare_actions: bool = False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = {"finetuned": checkpoint_path}
    if checkpoint2_path:
        checkpoints["base"] = checkpoint2_path

    for ckpt_name, ckpt_path in checkpoints.items():
        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt_name} ({ckpt_path})")
        print(f"{'='*60}")

        policy = load_policy(ckpt_path, device)

        # --- Collect hidden states for all instructions ---
        all_hidden_states = {}   # raw [1, seq, 2048] per instruction
        prefix_pooled = {}       # mean-pooled full prefix
        img_pooled = {}          # mean-pooled image tokens only
        lang_pooled = {}         # mean-pooled language tokens only
        action_outputs = {}      # action chunks (if requested)
        labels = {}
        num_img_tokens = None

        for instr in instructions:
            text = instr["text"]
            group = instr.get("group", "default")
            print(f"  Extracting: '{text}'")

            images, img_masks, tokens, masks = prepare_inputs(
                policy, image_path, text, device
            )

            hidden_states, prefix_embs, pad_masks, n_img = extract_vlm_hidden_states(
                policy, images, img_masks, tokens, masks
            )
            num_img_tokens = n_img

            # Pool: full prefix
            prefix_pooled[text] = pool_embeddings(
                hidden_states, pad_masks, method=pooling
            ).squeeze(0).cpu()

            # Pool: image region only [0 : num_img_tokens]
            img_pooled[text] = pool_region(
                hidden_states, pad_masks, 0, num_img_tokens, method=pooling
            ).squeeze(0).cpu()

            # Pool: language region only [num_img_tokens : end]
            lang_pooled[text] = pool_region(
                hidden_states, pad_masks, num_img_tokens, hidden_states.shape[1],
                method=pooling
            ).squeeze(0).cpu()

            # Keep raw hidden states for per-token analysis
            all_hidden_states[text] = hidden_states.cpu()

            labels[text] = group

            # Optional: action outputs
            if compare_actions:
                actions = extract_action_outputs(
                    policy, images, img_masks, tokens, masks
                )
                action_outputs[text] = actions.cpu()

        # =====================================================================
        # Analysis 1: Cosine similarity matrices
        # =====================================================================
        for region_name, region_dict in [
            ("full_prefix", prefix_pooled),
            ("image_tokens", img_pooled),
            ("language_tokens", lang_pooled),
        ]:
            print(f"\n--- Cosine similarities: {region_name} ({ckpt_name}) ---")
            results, sim_matrix, names = cosine_similarity_matrix(region_dict)

            for (ni, nj), sim in results.items():
                if ni < nj:
                    print(f"    {ni[:40]:40s}  vs  {nj[:40]:40s}  →  {sim:.4f}")

            plot_similarity_matrix(
                sim_matrix, names,
                f"{region_name} similarity ({ckpt_name})",
                output_dir / f"sim_{region_name}_{ckpt_name}.png"
            )

        # =====================================================================
        # Analysis 2: Per-token position similarity
        # =====================================================================
        print(f"\n--- Per-token analysis ({ckpt_name}) ---")
        instr_texts = [i["text"] for i in instructions]
        per_token_sims = []
        pair_labels = []

        # Compare each instruction against the first instruction
        ref_text = instr_texts[0]
        for other_text in instr_texts[1:]:
            sims = per_token_cosine_similarity(
                all_hidden_states[ref_text],
                all_hidden_states[other_text],
            )
            per_token_sims.append(sims)
            pair_labels.append(f"'{ref_text[:25]}' vs '{other_text[:25]}'")

            # Summary stats
            img_sim = sims[:num_img_tokens].mean().item()
            lang_sim = sims[num_img_tokens:].mean().item()
            print(f"    {ref_text[:30]:30s}  vs  {other_text[:30]:30s}")
            print(f"      Image tokens mean sim:    {img_sim:.4f}")
            print(f"      Language tokens mean sim:  {lang_sim:.4f}")
            print(f"      Delta (lang - img):        {lang_sim - img_sim:+.4f}")

        if per_token_sims:
            plot_per_token_similarity(
                per_token_sims, pair_labels, num_img_tokens,
                f"Per-token cosine similarity ({ckpt_name})",
                output_dir / f"per_token_{ckpt_name}.png"
            )

        # =====================================================================
        # Analysis 3: UMAP
        # =====================================================================
        if len(instructions) >= 3:
            plot_umap(
                prefix_pooled, labels,
                f"UMAP — full prefix ({ckpt_name})",
                output_dir / f"umap_prefix_{ckpt_name}.png"
            )
            plot_umap(
                img_pooled, labels,
                f"UMAP — image tokens only ({ckpt_name})",
                output_dir / f"umap_image_{ckpt_name}.png"
            )
            plot_umap(
                lang_pooled, labels,
                f"UMAP — language tokens only ({ckpt_name})",
                output_dir / f"umap_lang_{ckpt_name}.png"
            )

        # =====================================================================
        # Analysis 4: Action output comparison (optional)
        # =====================================================================
        if compare_actions and action_outputs:
            print(f"\n--- Action output comparison ({ckpt_name}) ---")
            action_texts = list(action_outputs.keys())
            for i, t1 in enumerate(action_texts):
                for t2 in action_texts[i + 1:]:
                    a1 = action_outputs[t1]
                    a2 = action_outputs[t2]
                    l2_dist = torch.norm(a1 - a2).item()
                    max_diff = torch.max(torch.abs(a1 - a2)).item()
                    mean_diff = torch.mean(torch.abs(a1 - a2)).item()
                    print(f"    '{t1[:35]}' vs '{t2[:35]}'")
                    print(f"      L2 dist: {l2_dist:.4f}  |  max joint diff: {max_diff:.4f}  |  mean diff: {mean_diff:.4f}")

        # =====================================================================
        # Save raw results
        # =====================================================================
        results_data = {
            "checkpoint": ckpt_path,
            "num_img_tokens": num_img_tokens,
            "pooling": pooling,
            "instructions": [i["text"] for i in instructions],
            "cosine_similarities": {
                region: {
                    f"{ni} ||| {nj}": sim
                    for (ni, nj), sim in cosine_similarity_matrix(d)[0].items()
                    if ni <= nj
                }
                for region, d in [
                    ("full_prefix", prefix_pooled),
                    ("image_tokens", img_pooled),
                    ("language_tokens", lang_pooled),
                ]
            },
        }

        if compare_actions and action_outputs:
            action_comparisons = {}
            action_texts = list(action_outputs.keys())
            for i, t1 in enumerate(action_texts):
                for t2 in action_texts[i + 1:]:
                    a1, a2 = action_outputs[t1], action_outputs[t2]
                    action_comparisons[f"{t1} ||| {t2}"] = {
                        "l2_dist": torch.norm(a1 - a2).item(),
                        "max_joint_diff": torch.max(torch.abs(a1 - a2)).item(),
                        "mean_diff": torch.mean(torch.abs(a1 - a2)).item(),
                    }
            results_data["action_comparisons"] = action_comparisons

        with open(output_dir / f"results_{ckpt_name}.json", "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"\n  Results saved to {output_dir / f'results_{ckpt_name}.json'}")

        del policy
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# 9. Example instructions
# --------------------------------------------------------------------------- #

EXAMPLE_INSTRUCTIONS = [
    # Preference variants of the same base task
    {"text": "pick up the red block and place it on the pink tray",      "group": "pick_place"},
    {"text": "pick up the blue block and place it on the pink tray",     "group": "pick_place"},
    {"text": "pick up the green block and place it on the pink tray",    "group": "pick_place"},

    # Spatial preference variants
    {"text": "pick up the block closest to the tray",                    "group": "spatial"},
    {"text": "pick up the block farthest from the tray",                 "group": "spatial"},

    # Semantically different tasks (control — should be far apart)
    {"text": "stack the blocks on top of each other",                    "group": "different_task"},
    {"text": "push the block to the left side of the table",             "group": "different_task"},

    # Minimal pair: same task with/without preference
    {"text": "pick up a block and place it on the pink tray",            "group": "no_preference"},
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PI05 VLM embedding analysis")
    parser.add_argument("--checkpoint", required=True,
                        help="Path or HF repo of finetuned checkpoint")
    parser.add_argument("--checkpoint2", default=None,
                        help="Optional second checkpoint (e.g. base) for comparison")
    parser.add_argument("--image", required=True,
                        help="Path to a saved camera frame (.png/.jpg)")
    parser.add_argument("--output", default="embedding_analysis",
                        help="Output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pooling", default="mean", choices=["mean", "last", "cls"])
    parser.add_argument("--compare-actions", action="store_true",
                        help="Also run full denoising and compare action outputs")
    parser.add_argument("--instructions-file", default=None,
                        help="JSON file with instruction list (keys: text, group)")

    args = parser.parse_args()

    if args.instructions_file:
        with open(args.instructions_file) as f:
            instructions = json.load(f)
    else:
        instructions = EXAMPLE_INSTRUCTIONS
        print("Using built-in example instructions. "
              "Pass --instructions-file for custom ones.\n")

    run_comparison(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        instructions=instructions,
        output_dir=args.output,
        device=args.device,
        pooling=args.pooling,
        checkpoint2_path=args.checkpoint2,
        compare_actions=args.compare_actions,
    )