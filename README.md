# Thesis: Dual-System VLM + VLA Control on SO-101

A dual-system robot control stack for the SO-101 arm: a vision-language model (Qwen3-VL) decomposes high-level tasks into subtasks, and a vision-language-action policy (SmolVLA / Pi0.5) executes them via [lerobot](https://github.com/huggingface/lerobot) async inference. Policies are evaluated both on real hardware and in simulation with [LeIsaac](https://github.com/LightwheelAI/leisaac) (Isaac Sim).

## Repository layout

| Path | Description |
| --- | --- |
| `lerobot_so101/` | Real-robot scripts: teleop, calibration, cameras, VLM planner, and the VLM→VLA orchestrator (`data/vlm_robot_orchestrator.py`) |
| `leisaac/` | LeIsaac simulation environment (Dockerized) |
| `policy_server/` | Dockerized lerobot policy server |
| `eval/` | Policy evaluation scripts (LIBERO benchmarks, custom inference) |
| `assets/` | Robot and scene assets for simulation |

---

## Data collection (real robot)

### Teleoperation

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=Kumquat_follower \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=Kumquat_leader \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: intelrealsense, serial_number_or_name: 342222071104, width: 640, height: 480, fps: 30}}" \
    --display_data=true
```

### Recording episodes

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=Kumquat_follower \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: intelrealsense, serial_number_or_name: 342222071104, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=Kumquat_leader \
    --display_data=true \
    --dataset.repo_id=ant0nh/pnp_orange_50 \
    --dataset.num_episodes=50 \
    --dataset.single_task="Grab the orange and put it in the bowl" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=15
```

### Merging datasets

```bash
lerobot-edit-dataset \
  --new_repo_id ant0nh/pnp_275 \
  --operation.type merge \
  --operation.repo_ids "['ant0nh/pnp_tray_75', 'ant0nh/pnp_200']"
```

---

## Training

Fine-tune Pi0.5 on a recorded dataset:

```bash
lerobot-train \
    --dataset.repo_id=ant0nh/pnp_350 \
    --policy.type=pi05 \
    --output_dir=./outputs/pi05 \
    --job_name=pi05_training \
    --policy.repo_id=ant0nh/pi05_pnp_350_35k \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --steps=35000 \
    --policy.device=cuda \
    --batch_size=32 \
    --save_freq=35000
```

---

## Inference (real robot)

### Lerobot async inference

Start the policy server:

```bash
python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080
```

Then run the robot client loop:

```bash
python robot_client_loop.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=arm \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: intelrealsense, serial_number_or_name: 342222071104, width: 640, height: 480, fps: 30}}" \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=pi05 \
    --pretrained_name_or_path=ant0nh/pi05_275_28k \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.7 \
    --episode_duration=30
```

### Rollout with real-time chunking (2 cameras)

```bash
lerobot-rollout \
    --strategy.type=base \
    --policy.path=ant0nh/pi05_130 \
    --inference.type=rtc \
    --inference.rtc.execution_horizon=10 \
    --inference.rtc.max_guidance_weight=10.0 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: intelrealsense, serial_number_or_name: 342222071104, width: 640, height: 480, fps: 30}}" \
    --task="pick up the orange" \
    --duration=60
```

> **Note:** Pi models expect `left_wrist_0_rgb` and `base_0_rgb` as the default camera names.

---

## Inference (simulation, LeIsaac)

### SmolVLA via lerobot policy server

```bash
python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-SO101-PickOrange-v0 \
    --policy_type=lerobot-smolvla \
    --policy_host=localhost \
    --policy_port=8080 \
    --policy_timeout_ms=5000 \
    --policy_language_instruction='Pick up an orange' \
    --policy_checkpoint_path=edge-inference/smolvla-so101-pick-orange \
    --policy_action_horizon=50 \
    --device=cuda \
    --enable_cameras
```

Use `--policy_checkpoint_path=lerobot/smolvla_base` to evaluate the base (non-finetuned) SmolVLA model.

### Openpi

Serve a Pi0.5 checkpoint with openpi:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_so101_pick_orange \
  --policy.dir=/app/checkpoints/pi05_pnp_orange/20000
```

Run inference against it from LeIsaac:

```bash
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

---

## Troubleshooting

**Rerun / X11 auth inside Docker** — grant the container access to the host display:

```bash
XAUTH=/tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
chmod 777 $XAUTH
```
