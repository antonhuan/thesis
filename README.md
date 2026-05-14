Startup guide for simulator
xhost +local:docker
# Inference in Leisaac
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

python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --policy_type=lerobot-smolvla \
    --policy_host=localhost \
    --policy_port=8080 \
    --policy_timeout_ms=5000 \
    --policy_language_instruction='Pick up an orange' \
    --policy_checkpoint_path=lerobot/smolvla_base\
    --policy_action_horizon=50 \
    --device=cuda \
    --enable_cameras

Changes made to base image
-Patch helper to rename cameras to the expected names for policy 
-Added side camera to single_arm_env.py at pos=(0.72684, -0.22668, 0.14343), rot=(-0.5, 0.5, 0.5, -0.5)
-Patched torch_dtype in opt/venv/lib/python3.12/site-packages/lerobot/policies/smolvla/smolvlm_with_expert.py to dtype

To clear cache 
```
find /workspace/leisaac -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
```

# Teleop command
```
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=Kumquat_follower \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=Kumquat_leader \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --display_data=true
```
# Record data
```
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=Kumquat_follower \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=Kumquat_leader \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/record-test \
    --dataset.num_episodes=5 \
    --dataset.single_task="Grab the bag" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=20
```

# Rollout 2 cams

```
lerobot-rollout \
    --strategy.type=base \
    --policy.path=ant0nh/smolvla_finetuned \
    --inference.type=rtc \
    --inference.rtc.execution_horizon=10 \
    --inference.rtc.max_guidance_weight=10.0 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: intelrealsense, serial_number_or_name: 342222071104, width: 640, height: 480, fps: 30}}" \
    --task="pick up the orange and put it in the bowl" \
    --duration=60
```
# Rollout 1 cam

```
lerobot-rollout \
    --strategy.type=base \
    --policy.path=edge-inference/smolvla-so101-pick-orange \
    --inference.type=rtc \
    --inference.rtc.execution_horizon=10 \
    --inference.rtc.max_guidance_weight=10.0 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --task="pick up the orange" \
    --duration=60 
    
```
# Openpi 
```
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_so101_pick_orange \
  --policy.dir=/app/checkpoints/pi05_pnp_orange/20000
```

```
python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --policy_type=openpi \
    --policy_host=10.0.0.1 \
    --policy_port=8000 \
    --policy_timeout_ms=5000 \
    --policy_language_instruction='Pick the orange to the plate' \
    --device=cuda \
    --enable_cameras
```
# Xauth for rerun 
```
XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
chmod 777 $XAUTH
```