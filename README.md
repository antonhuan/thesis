Startup guide for simulator

xhost +local:docker
```
python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --policy_type=lerobot-smolvla \
    --policy_host=localhost \
    --policy_port=8080 \
    --policy_timeout_ms=5000 \
    --policy_language_instruction='Pick up an orange' \
    --policy_checkpoint_path=edge-inference/smolvla-so101-pick-orange\
    --policy_action_horizon=50 \
    --device=cuda \
    --enable_cameras
```

Changes made to base image
-Patch helper to rename cameras to the expected names for policy 
-Added side camera to single_arm_env.py at pos=(0.72684, -0.22668, 0.14343), rot=(-0.5, 0.5, 0.5, -0.5)
-Patched torch_dtype in opt/venv/lib/python3.12/site-packages/lerobot/policies/smolvla/smolvlm_with_expert.py to dtype

To clear cache 
```
find /workspace/leisaac -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
```
