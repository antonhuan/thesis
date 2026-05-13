import numpy as np
import time
from openpi_client import image_tools, websocket_client_policy
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

config = SOFollowerRobotConfig(
    port="/dev/ttyACM0",
    max_relative_target=5.0,
    cameras={
        "wrist": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
        "front": RealSenseCameraConfig(
            serial_number_or_name="342222071104", width=640, height=480, fps=30
        ),
    },
)

robot = SOFollower(config)
robot.connect()

client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

try:
    for step in range(600):
        obs = robot.get_observation()

        state = np.array([
            obs["shoulder_pan.pos"],
            obs["shoulder_lift.pos"],
            obs["elbow_flex.pos"],
            obs["wrist_flex.pos"],
            obs["wrist_roll.pos"],
            obs["gripper.pos"],
        ], dtype=np.float32)

        observation = {
            "images/front": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["front"], 224, 224)
            ),
            "images/wrist": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["wrist"], 224, 224)
            ),
            "state": state,
            "prompt": "pick up the orange",
        }

        action_chunk = client.infer(observation)["actions"]

        action = action_chunk[0]
        robot.send_action({
            "shoulder_pan.pos": float(action[0]),
            "shoulder_lift.pos": float(action[1]),
            "elbow_flex.pos": float(action[2]),
            "wrist_flex.pos": float(action[3]),
            "wrist_roll.pos": float(action[4]),
            "gripper.pos": float(action[5]),
        })

        time.sleep(0.1)

finally:
    robot.disconnect()
