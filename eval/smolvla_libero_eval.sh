#!/bin/bash
set -e

echo "Starting lerobot eval..."
echo n | lerobot-eval \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --policy.compile_model=false \
  --policy.push_to_hub=false \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids=[0] \
  --env.render_mode=human \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --policy.n_action_steps=10 \
  --output_dir=/workspace/eval_logs/smolvla
