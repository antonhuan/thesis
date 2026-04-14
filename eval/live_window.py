import sys
from unittest.mock import MagicMock

# lerobot/policies/__init__.py eagerly imports groot, which has a broken dataclass
# in the current upstream. Mock it out so the rest of lerobot loads fine.
for _mod in [
    "lerobot.policies.groot",
    "lerobot.policies.groot.configuration_groot",
    "lerobot.policies.groot.modeling_groot",
    "lerobot.policies.groot.groot_n1",
]:
    sys.modules[_mod] = MagicMock()

# transformers renamed is_flash_attn_greater_or_equal_2_10 -> is_flash_attn_greater_or_equal
# but lerobot's modeling_florence2 still imports the old name.
import transformers.utils as _tu
if not hasattr(_tu, "is_flash_attn_greater_or_equal_2_10"):
    _tu.is_flash_attn_greater_or_equal_2_10 = _tu.is_flash_attn_greater_or_equal

import gym
import mujoco
import mujoco.viewer
import torch
import importlib

from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.utils import get_device_from_parameters

# Set up env
importlib.import_module("libero2gym")
env = gym.make("libero-spatial-v0", task_id=0)

# Load X-VLA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = XVLAPolicy.from_pretrained("lerobot/xvla-libero").to(device).eval()

# Launch the MuJoCo passive viewer for live rendering
viewer = mujoco.viewer.launch_passive(
    env.unwrapped.model, env.unwrapped.data
)

observation, info = env.reset(seed=42)
viewer.sync()

# YOUR custom language instruction
custom_task = "pick up the red bowl and place it on the plate"

for i in range(2000):
    observation = preprocess_observation(observation)
    observation = {
        key: observation[key].to(device, non_blocking=True)
        for key in observation
    }
    
    # Inject your custom language prompt here
    observation["task"] = custom_task
    
    with torch.inference_mode():
        action = policy.select_action(observation)
    
    action = action.to("cpu").numpy()
    observation, reward, terminated, truncated, info = env.step(action[0])
    viewer.sync()  # update the live viewer
    
    if terminated or truncated:
        observation, info = env.reset()
        viewer.sync()

viewer.close()
env.close()