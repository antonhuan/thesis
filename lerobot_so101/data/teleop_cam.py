from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader
from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower
from lerobot.cameras.configs import ColorMode, Cv2Rotation


camera_config = {
    "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30)
}

robot_config = SO101FollowerConfig(
    port="/dev/ttyACM1",
    id="Kumquat_follower",
    cameras=camera_config, 

)

teleop_config = SO101LeaderConfig(
    port="/dev/ttyACM0",
    id="Kumquat_leader",
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)
robot.connect()
teleop_device.connect()

while True:
    observation = robot.get_observation()
    action = teleop_device.get_action()
    robot.send_action(action)