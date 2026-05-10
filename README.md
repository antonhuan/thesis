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
#Teleop command
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
#Record data
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