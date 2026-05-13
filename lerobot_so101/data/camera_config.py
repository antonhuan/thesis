from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

# Construct an `OpenCVCameraConfig` with your desired FPS, resolution, color mode, and rotation.
config = OpenCVCameraConfig(
    index_or_path=0,
    fps=30,
    width=640,
    height=480,
    color_mode=ColorMode.RGB,
    rotation=Cv2Rotation.NO_ROTATION
)

# Instantiate and connect an `OpenCVCamera`, performing a warm-up read (default).
with OpenCVCamera(config) as camera:

    # Read a frame synchronously — blocks until hardware delivers a new frame
    frame = camera.read()
    print(f"read() call returned frame with shape:", frame.shape)

    # Read a frame asynchronously with a timeout — returns the latest unconsumed frame or waits up to timeout_ms for a new one
    try:
        for i in range(10):
            frame = camera.async_read(timeout_ms=200)
            print(f"async_read call returned frame {i} with shape:", frame.shape)
    except TimeoutError as e:
        print(f"No frame received within timeout: {e}")

    # Instantly return a frame - returns the most recent frame captured by the camera
    try:
        initial_frame = camera.read_latest(max_age_ms=1000)
        for i in range(10):
            frame = camera.read_latest(max_age_ms=1000)
            print(f"read_latest call returned frame {i} with shape:", frame.shape)
            print(f"Was a new frame received by the camera? {not (initial_frame == frame).any()}")
    except TimeoutError as e:
        print(f"Frame too old: {e}")
