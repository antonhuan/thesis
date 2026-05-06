from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower

config = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="Kumquat_follower",
)
follower = SO101Follower(config)
follower.connect(calibrate=False)
follower.calibrate()
follower.disconnect()
# from lerobot.teleoperators.so_leader import SO101LeaderConfig, SO101Leader

# config = SO101LeaderConfig(
#     port="/dev/ttyACM0",
#     id="Kumquat_leader",
# )

# leader = SO101Leader(config)
# leader.connect(calibrate=False)
# leader.calibrate()
# leader.disconnect()