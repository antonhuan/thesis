#!/bin/bash
set -e

echo "Starting eval"
echo n | lerobot-eval \
  --policy.path=lerobot/xvla-libero \
  --env.type=libero \
  --env.task=libero_object \
  --env.control_mode=absolute \
  --env.render_mode=human \
  --env.episode_length=800 \
  --eval.batch_size=1 \
  --eval.n_episodes=1 \
  --output_dir=/workspace/eval_logs/xvla