"""
Interactive VLM final-hidden-state comparison for a finetuned PI05 model.

Pulls a live frame from the RealSense camera, asks for two text instructions,
and analyses whether the VLM backbone's final representation — the KV cache
boundary that conditions the action expert — separates the two prompts or
collapses them. Then loops.

Architecture target:
    PI05Policy.model.paligemma_with_expert.paligemma.model.language_model
    → forward() → .last_hidden_state  [B, seq_len, 2048]

    This is the output of the Gemma 2B VLM after 18 transformer layers
    with prefix-LM attention over [256 image tokens | N language tokens].
    It gets cached as KV pairs and served to the action expert during
    all denoising steps — if two instructions produce identical states here,
    the action expert has zero information to differentiate them.

Per iteration:
    1. Capture a live top-down frame from the RealSense D435.
    2. Prompt for two instructions.
    3. Cosine similarity between the two (full prefix / image tokens / language tokens).
    4. Per-token position analysis (image vs language token regions), saved as a plot.
    5. UMAP projection of per-token hidden states, saved as a plot.

Usage:
    python vla_embedding_comp.py --checkpoint /path/to/finetuned/checkpoint \
        --output embedding_analysis

Interactive commands:
    /quit, /exit   exit
    /help          show help
    anything else  enter it at the "prompt A" / "prompt B" prompts

Requires the RealSense SDK (pyrealsense2) and the RealSense D435 connected.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Reuse the RealSense capture helpers from the interactive VLM REPL (same dir).
from vlm import RealSenseCamera, capture_scene


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

def prepare_inputs(policy, image, instruction: str, device: str = "cuda"):
    """
    Convert a live PIL frame + instruction string into the tensors that
    PI05Pytorch.embed_prefix() expects: (images, img_masks, tokens, masks).
    """
    from lerobot.policies.pi05.modeling_pi05 import resize_with_pad_torch
    from transformers import AutoTokenizer

    config = policy.config

    # --- Image ---
    img = image.convert("RGB")
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
        max_length=config.tokenizer_max_length,
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
    vlm = model.paligemma_with_expert.paligemma.model.language_model
    # Build attention masks and position IDs
    att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    att_2d_masks_4d = att_2d_masks[:, None, :, :]
    model_dtype = next(vlm.parameters()).dtype
    att_2d_masks_4d = torch.where(att_2d_masks_4d, 
                                torch.tensor(0.0, dtype=model_dtype, device=att_2d_masks_4d.device),
                                torch.tensor(-1e9, dtype=model_dtype, device=att_2d_masks_4d.device))

    # Forward through VLM backbone only
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
# 4. Pooling utilities
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
# 5. Analysis functions
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
    return (a * b).sum(dim=-1).float().cpu()  # [seq]


# --------------------------------------------------------------------------- #
# 6. Plotting
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
    Plot per-position cosine similarity for one or more instruction pairs.
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


def plot_umap_tokens(hidden_a, hidden_b, pad_masks_a, pad_masks_b, num_img_tokens, label_a, label_b, save_path):
    """
    UMAP projection of per-token VLM hidden states from two prompts.
    Filters out padding tokens. Colors by prompt, shapes by region (image vs language).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from umap import UMAP

    a = hidden_a.squeeze(0).float().cpu().numpy()
    b = hidden_b.squeeze(0).float().cpu().numpy()
    mask_a = pad_masks_a.squeeze(0).cpu().numpy().astype(bool)
    mask_b = pad_masks_b.squeeze(0).cpu().numpy().astype(bool)

    seq_len = a.shape[0]

    # Build per-token metadata before filtering
    tokens_list = []
    prompts = []
    regions = []

    for i in range(seq_len):
        if mask_a[i]:
            tokens_list.append(a[i])
            prompts.append("A")
            regions.append("image" if i < num_img_tokens else "language")
    for i in range(seq_len):
        if mask_b[i]:
            tokens_list.append(b[i])
            prompts.append("B")
            regions.append("image" if i < num_img_tokens else "language")

    all_tokens = np.array(tokens_list)
    prompts = np.array(prompts)
    regions = np.array(regions)

    reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    projected = reducer.fit_transform(all_tokens)

    fig, ax = plt.subplots(figsize=(10, 8))

    combos = [
        ("A", "image",    "tab:blue",   "o", 0.4, 15),
        ("A", "language", "tab:blue",   "x", 0.8, 30),
        ("B", "image",    "tab:orange", "o", 0.4, 15),
        ("B", "language", "tab:orange", "x", 0.8, 30),
    ]

    labels_map = {"A": label_a[:35], "B": label_b[:35]}

    for prompt, region, color, marker, alpha, size in combos:
        mask = (prompts == prompt) & (regions == region)
        if not mask.any():
            continue
        label = f"{prompt} {region} ({labels_map[prompt]})" if region == "image" else f"{prompt} {region}"
        ax.scatter(projected[mask, 0], projected[mask, 1],
                   c=color, marker=marker, alpha=alpha, s=size, label=label)

    ax.legend(fontsize=7, loc='best', markerscale=1.5)
    ax.set_title('UMAP of per-token VLM hidden states (padding filtered)', fontsize=11)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# --------------------------------------------------------------------------- #
# 7. Two-prompt comparison
# --------------------------------------------------------------------------- #

def compare_pair(policy, image, prompt_a, prompt_b, output_dir, pooling="mean", device="cuda"):
    """
    Compare the VLM final hidden states of two instructions on a single frame.
    Prints cosine similarities + per-token summary and saves plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix_pooled = {}   # mean-pooled full prefix, keyed by prompt text
    img_pooled = {}      # mean-pooled image tokens only
    lang_pooled = {}     # mean-pooled language tokens only
    all_hidden = {}      # raw [1, seq, 2048]
    all_pad_masks = {}   # raw [1, seq] pad masks
    num_img_tokens = None

    for text in (prompt_a, prompt_b):
        print(f"  Extracting: '{text}'")
        images, img_masks, tokens, masks = prepare_inputs(policy, image, text, device)
        hidden_states, _prefix_embs, pad_masks, n_img = extract_vlm_hidden_states(
            policy, images, img_masks, tokens, masks
        )
        num_img_tokens = n_img

        prefix_pooled[text] = pool_embeddings(
            hidden_states, pad_masks, method=pooling
        ).squeeze(0).cpu()
        img_pooled[text] = pool_region(
            hidden_states, pad_masks, 0, num_img_tokens, method=pooling
        ).squeeze(0).cpu()
        lang_pooled[text] = pool_region(
            hidden_states, pad_masks, num_img_tokens, hidden_states.shape[1], method=pooling
        ).squeeze(0).cpu()
        all_hidden[text] = hidden_states.cpu()
        all_pad_masks[text] = pad_masks.cpu()

    # --- Cosine similarities (the single off-diagonal value per region) ---
    print("\n--- Cosine similarities ---")
    region_matrices = {}
    for region_name, region_dict in [
        ("full_prefix", prefix_pooled),
        ("image_tokens", img_pooled),
        ("language_tokens", lang_pooled),
    ]:
        results, sim_matrix, names = cosine_similarity_matrix(region_dict)
        region_matrices[region_name] = (sim_matrix, names)
        sim = results[(prompt_a, prompt_b)]
        print(f"    {region_name:16s} →  {sim:.4f}")

    # --- Per-token position analysis ---
    print("\n--- Per-token analysis ---")
    sims = per_token_cosine_similarity(all_hidden[prompt_a], all_hidden[prompt_b])
    img_sim = sims[:num_img_tokens].mean().item()
    lang_sim = sims[num_img_tokens:].mean().item()
    print(f"    Image tokens mean sim:     {img_sim:.4f}")
    print(f"    Language tokens mean sim:  {lang_sim:.4f}")
    print(f"    Delta (lang - img):        {lang_sim - img_sim:+.4f}")

    # --- Save plots (timestamped) ---
    ts = time.strftime("%Y%m%d_%H%M%S")
    full_matrix, full_names = region_matrices["full_prefix"]
    plot_similarity_matrix(
        full_matrix, full_names,
        "full_prefix similarity",
        output_dir / f"sim_{ts}.png",
    )
    plot_per_token_similarity(
        [sims],
        [f"'{prompt_a[:25]}' vs '{prompt_b[:25]}'"],
        num_img_tokens,
        "Per-token cosine similarity",
        output_dir / f"per_token_{ts}.png",
    )
    plot_umap_tokens(
        all_hidden[prompt_a], all_hidden[prompt_b],
        all_pad_masks[prompt_a], all_pad_masks[prompt_b],
        num_img_tokens,
        prompt_a, prompt_b,
        output_dir / f"umap_{ts}.png",
    )


# --------------------------------------------------------------------------- #
# 8. Interactive loop
# --------------------------------------------------------------------------- #

INTERACTIVE_HELP = """
Commands:
  /help          show this help
  /quit, /exit   exit

Otherwise: each iteration captures a fresh camera frame, then prompts you for
two instructions ("prompt A" and "prompt B") and compares the VLM hidden states
they produce on that frame. Blank input re-captures and re-prompts.
"""


def _read_prompt(label):
    """Read one instruction, handling /commands. Returns text, or None to exit,
    or "" to skip/re-loop."""
    while True:
        text = input(f"{label}> ").strip()
        if not text:
            return ""
        low = text.lower()
        if low in ("/quit", "/exit"):
            return None
        if low == "/help":
            print(INTERACTIVE_HELP)
            continue
        return text


def interactive_loop(policy, camera, output_dir, pooling="mean", device="cuda", save_frames=False):
    print(INTERACTIVE_HELP)
    frame_dir = Path(output_dir) / "frames" if save_frames else None

    while True:
        try:
            print("\n" + "=" * 60)
            print("Capturing frame...")
            image = capture_scene(camera, save_dir=frame_dir)

            prompt_a = _read_prompt("prompt A")
            if prompt_a is None:
                break
            if not prompt_a:
                continue

            prompt_b = _read_prompt("prompt B")
            if prompt_b is None:
                break
            if not prompt_b:
                continue

            compare_pair(policy, image, prompt_a, prompt_b, output_dir, pooling, device)
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break


# --------------------------------------------------------------------------- #
# 9. Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Interactive PI05 VLM embedding comparison (live camera, two prompts)"
    )
    parser.add_argument("--checkpoint", default="ant0nh/pi05_500_30k",
                        help="Path or HF repo of the checkpoint to analyse")
    parser.add_argument("--output", default="embedding_analysis",
                        help="Output directory for saved plots (and frames)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pooling", default="mean", choices=["mean", "last", "cls"])
    parser.add_argument("--resolution", default="640x480",
                        help="Camera resolution, WIDTHxHEIGHT (default 640x480)")
    parser.add_argument("--save-frames", action="store_true",
                        help="Save each captured frame to <output>/frames/")
    args = parser.parse_args()

    width, height = (int(x) for x in args.resolution.lower().split("x"))

    print(f"Loading policy from {args.checkpoint} ...")
    policy = load_policy(args.checkpoint, args.device)

    camera = RealSenseCamera(width=width, height=height)
    try:
        camera.start()
    except Exception as e:
        print(f"Failed to start RealSense camera: {e}")
        return

    try:
        interactive_loop(
            policy, camera, args.output,
            pooling=args.pooling, device=args.device, save_frames=args.save_frames,
        )
    finally:
        camera.stop()


if __name__ == "__main__":
    main()