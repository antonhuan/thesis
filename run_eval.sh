#!/usr/bin/env bash
# run_eval.sh — sets up X11 access and launches the lerobot eval container
set -euo pipefail

# ── HuggingFace token ────────────────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Error: HF_TOKEN is not set."
  echo "Export it before running: export HF_TOKEN=hf_..."
  exit 1
fi

# ── X11: allow the container to open a window on your desktop ────────────────
echo "Granting X11 access to Docker containers..."
xhost +local:docker

# ── Build if image doesn't exist ─────────────────────────────────────────────
if ! docker image inspect lerobot-eval:latest &>/dev/null; then
  echo "Image not found, building..."
  docker compose build
fi

# ── Log in to HuggingFace inside the container (first run only) ──────────────
# The eval command auto-downloads the model; HUGGING_FACE_HUB_TOKEN is picked
# up by huggingface_hub automatically from the environment.
echo "Starting eval (live simulation window should appear shortly)..."
docker compose up --remove-orphans

# ── Cleanup: revoke X11 access when done ─────────────────────────────────────
xhost -local:docker
echo "Done. Eval logs saved to ./eval_logs/"
