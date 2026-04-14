"""
Test script for LeIsaac environment with random actions.
Run this inside the container to verify the setup works.
"""
import torch
from lerobot.envs.factory import make_env

# Load the SO101 pick orange environment from the hub
envs_dict = make_env(
    "LightwheelAI/leisaac_env:envs/so101_pick_orange.py",
    n_envs=1,
    trust_remote_code=True,
)

# Access the environment
suite_name = next(iter(envs_dict))
sync_vector_env = envs_dict[suite_name][0]
env = sync_vector_env.envs[0].unwrapped

# Reset and run random actions
obs, info = env.reset()
print(f"Observation keys: {obs.keys() if hasattr(obs, 'keys') else type(obs)}")

step = 0
while True:
    action = torch.tensor(env.action_space.sample())
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1

    if step % 100 == 0:
        print(f"Step {step}, reward: {reward}")

    if terminated or truncated:
        print(f"Episode ended at step {step}")
        obs, info = env.reset()
        step = 0

env.close() 
