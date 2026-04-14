"""
custom_inference.py — X-VLA evaluation on LIBERO with custom instructions.
"""
import os
import cv2
import torch
import numpy as np
from gymnasium.vector import SyncVectorEnv
from lerobot.envs.libero import create_libero_envs
from lerobot.envs.utils import preprocess_observation, add_envs_task
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

def flatten_robot_state(obs):
    """Convert nested robot_state dict to flat observation.state tensor (8,)."""
    rs = obs.pop("observation.robot_state")
    joints = rs["joints"]["pos"]    # (n_envs, 7)
    gripper = rs["gripper"]["qpos"]  # (n_envs, 2)
    import torch
    if not isinstance(joints, torch.Tensor):
        joints = torch.tensor(joints, dtype=torch.float32)
        gripper = torch.tensor(gripper, dtype=torch.float32)
    obs["observation.state"] = torch.cat([joints, gripper[..., :1]], dim=-1)
    return obs

HEADLESS = os.environ.get("HEADLESS", "0") == "1"

CUSTOM_INSTRUCTIONS = [
    "grab the soup can and drop it in the basket",
    "take the alphabet soup and put it in the container",
    "move the can into the basket",
]
TASK_SUITE = "libero_object"
EPISODE_LENGTH = 800
OUTPUT_DIR = "/workspace/eval_logs"
MODEL_ID = "lerobot/xvla-libero"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load policy + processors ────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading policy on {device}...")
policy = XVLAPolicy.from_pretrained(MODEL_ID).to(device).eval()

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    MODEL_ID,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)
print("Policy and processors loaded.")

# ── Create envs ─────────────────────────────────────────
print(f"Creating envs for suite: {TASK_SUITE}")
envs = create_libero_envs(
    task=TASK_SUITE,
    n_envs=1,
    control_mode="absolute",
    episode_length=EPISODE_LENGTH,
    env_cls=SyncVectorEnv,
    gym_kwargs={"obs_type": "pixels_agent_pos"}
)

# ── Run eval ─────────────────────────────────────────────
results = {}
for suite_name, task_map in envs.items():
    for task_id, vec_env in task_map.items():
        for instruction in CUSTOM_INSTRUCTIONS:
            print(f"\n{'='*60}")
            print(f"Task ID  : {task_id}")
            print(f"Instruction: '{instruction}'")
            print(f"{'='*60}")

            # patch instruction on all sub-envs
            for env in vec_env.envs:
                env.task_description = instruction

            obs, _ = vec_env.reset()
            obs = preprocess_observation(obs)
            obs = add_envs_task(vec_env, obs)
            obs = flatten_robot_state(obs)
            batch = preprocess(obs)

            done = False
            step = 0
            while not done:
                # inference — returns full chunk (n_envs, n_action_steps, action_dim)
                with torch.no_grad():
                    action_chunk = policy.select_action(batch)
                    action_chunk = postprocess(action_chunk)

                # execute each action in the chunk
                for a_idx in range(action_chunk.shape[1]):
                    if done:
                        break

                    frame = vec_env.envs[0].render()
                    if HEADLESS:
                        if step % 100 == 0:
                            safe_instr = instruction.replace(" ", "_")[:40]
                            path = os.path.join(
                                OUTPUT_DIR,
                                f"task{task_id}_{safe_instr}_step{step:04d}.png",
                            )
                            cv2.imwrite(path, frame)
                    else:
                        cv2.imshow("X-VLA Eval", frame)
                        cv2.waitKey(1)

                    action = action_chunk[:, a_idx]  # (n_envs, action_dim)
                    obs, reward, terminated, truncated, info = vec_env.step(
                        action.cpu().numpy()
                    )
                    done = np.any(terminated) or np.any(truncated)
                    step += 1
                    if step % 50 == 0:
                        print(f"  Step {step}...")

                # update obs for next inference call
                obs = preprocess_observation(obs)
                obs = add_envs_task(vec_env, obs)
                obs = flatten_robot_state(obs)
                batch = preprocess(obs)

            success = info[0].get("is_success", False)
            print(f"Result: {'SUCCESS' if success else 'FAILURE'} after {step} steps")
            results[instruction] = success

if not HEADLESS:
    cv2.destroyAllWindows()

# ── Summary ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"{'='*60}")
for instruction, success in results.items():
    status = "success" if success else "fail"
    print(f"  {status} {instruction}")

total = len(results)
passed = sum(results.values())
print(f"\nSuccess rate: {passed}/{total} ({100*passed/total:.1f}%)")