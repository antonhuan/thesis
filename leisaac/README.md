Startup guide for simulator
```
python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --policy_type=lerobot-smolvla \
    --policy_host=localhost \
    --policy_port=8080 \
    --policy_timeout_ms=5000 \
    --policy_language_instruction='Pick the orange to the plate' \
    --policy_checkpoint_path=lerobot/smolvla_base\
    --policy_action_horizon=50 \
    --device=cuda \
    --enable_cameras
```