# robot_client_loop.py
#
# Modified from lerobot/async_inference/robot_client.py
# Adds a persistent task loop: the policy server connection and model stay
# loaded, and between episodes the arm returns to its home position and
# waits for the next task prompt via stdin.
#
# Usage:
#   Terminal 1 (policy server, stays running):
#     python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080
#
#   Terminal 2 (this script):
#     python robot_client_loop.py \
#       --robot.type=so101_follower \
#       --robot.port=/dev/ttyACM0 \
#       --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: intelrealsense, serial_number_or_name: 342222071104, width: 640, height: 480, fps: 30}}" \
#       --task="initial_task" \
#       --server_address=127.0.0.1:8080 \
#       --policy_type=pi05 \
#       --pretrained_name_or_path=ant0nh/pi05_pnp_425_25k \
#       --policy_device=cuda \
#       --actions_per_chunk=50 \
#       --chunk_size_threshold=0.7 \
#       --episode_duration=20
#
# At the prompt, type a task instruction and press Enter to execute.
# Type 'quit' or 'exit' to shut down cleanly.

import logging
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pprint import pformat
from queue import Queue, Empty
from typing import Any

import draccus
import grpc
import numpy as np
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.import_utils import register_third_party_plugins

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)


# ---------------------------------------------------------------------------
# Extended config: adds episode_duration and home_steps
# ---------------------------------------------------------------------------
@dataclass
class LoopClientConfig(RobotClientConfig):
    """Extends the base config with episode duration and homing parameters."""
    # Max seconds per episode before auto-termination
    episode_duration: float = 30
    # Number of interpolation steps when returning to home position
    home_steps: int = 50
    # Seconds to sleep between interpolation steps during homing
    home_step_dt: float = 0.04
    # Number of consecutive similar actions before declaring convergence
    convergence_window: int = 25
    # L2 threshold: actions closer than this are "similar"
    convergence_threshold: float = 1
    # Don't check convergence in the first N seconds (let the arm start moving)
    convergence_grace_period: float = 6
    # Seconds with no new actions before declaring a stall (policy stopped sending)
    action_stall_timeout: float = 2.0


@dataclass
class EpisodeResult:
    """Returned by run_episode to let the orchestrator know what happened."""
    duration: float
    converged: bool
    max_displacement: float
    total_actions: int


# ---------------------------------------------------------------------------
# The persistent-loop robot client
# ---------------------------------------------------------------------------
class LoopRobotClient:
    """
    Wraps the core async inference client logic with an outer loop that:
      1. Connects to the policy server and loads the model (once)
      2. Captures the arm's resting position as "home"
      3. Waits for a task prompt from stdin
      4. Runs the async control loop for --episode_duration seconds
      5. Returns the arm to home
      6. Goes back to step 3
    """

    prefix = "loop_client"
    logger = get_logger(prefix)

    def __init__(self, config: LoopClientConfig):
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        self.lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            self.lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )

        self.channel = grpc.insecure_channel(
            self.server_address,
            grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s"),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        # Episode-level state
        self.shutdown_event = threading.Event()
        self.episode_done = threading.Event()

        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1
        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(3)

        self.fps_tracker = FPSTracker(target_fps=self.config.fps)
        self.must_go = threading.Event()
        self.must_go.set()

        # Convergence tracking (initialised properly in _reset_episode_state)
        self._action_history = []
        self._first_action = None
        self._max_displacement = 0.0
        self._converged = False
        self._total_actions = 0
        self._last_action_time = 0.0

        # Capture home position on startup
        self.home_position = self._capture_current_position()
        self.logger.info(f"Home position captured: {self.home_position}")

        self.logger.info("Robot connected and ready")

    # ------------------------------------------------------------------
    # Home position helpers
    # ------------------------------------------------------------------
    def _capture_current_position(self) -> dict[str, float]:
        """Read the current joint positions as the home/rest pose."""
        obs = self.robot.get_observation()
        home = {}
        for key in self.robot.action_features:
            # observation keys are like "shoulder_pan.pos", action keys are "shoulder_pan"
            obs_key = f"{key}.pos" if f"{key}.pos" in obs else key
            if obs_key in obs:
                home[key] = float(obs[obs_key])
            else:
                # Fallback: try to find the key directly
                home[key] = 0.0
                self.logger.warning(f"Could not find observation key for {key}, defaulting to 0.0")
        return home

    def go_home(self):
        """Smoothly interpolate the arm back to its home position."""
        self.logger.info("Returning to home position...")
        current = self._capture_current_position()

        for step in range(1, self.config.home_steps + 1):
            alpha = step / self.config.home_steps
            interpolated = {}
            for key in self.robot.action_features:
                start_val = current.get(key, 0.0)
                end_val = self.home_position.get(key, 0.0)
                interpolated[key] = start_val + alpha * (end_val - start_val)
            self.robot.send_action(interpolated)
            time.sleep(self.config.home_step_dt)

        self.logger.info("Home position reached.")

    # ------------------------------------------------------------------
    # Server handshake (one-time)
    # ------------------------------------------------------------------
    def connect_server(self) -> bool:
        """Perform the gRPC handshake and send policy config to load the model."""
        try:
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            elapsed = time.perf_counter() - start_time
            self.logger.debug(f"Connected to policy server in {elapsed:.4f}s")

            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.info(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )
            self.stub.SendPolicyInstructions(policy_setup)
            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    # ------------------------------------------------------------------
    # Observation / action plumbing (mirrors original robot_client.py)
    # ------------------------------------------------------------------
    def send_observation(self, obs: TimedObservation) -> bool:
        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            return True
        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation: {e}")
            return False

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        if aggregate_fn is None:
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {
            action.get_timestep(): action.get_action() for action in internal_queue
        }

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            if new_action.get_timestep() <= latest_action:
                continue
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
            else:
                future_action_queue.put(
                    TimedAction(
                        timestamp=new_action.get_timestamp(),
                        timestep=new_action.get_timestep(),
                        action=aggregate_fn(
                            current_action_queue[new_action.get_timestep()],
                            new_action.get_action(),
                        ),
                    )
                )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        return {
            key: action_tensor[i].item()
            for i, key in enumerate(self.robot.action_features)
        }

    def _ready_to_send_observation(self) -> bool:
        with self.action_queue_lock:
            return self.action_queue.qsize() / max(self.action_chunk_size, 1) <= self._chunk_size_threshold

    # ------------------------------------------------------------------
    # Episode state management
    # ------------------------------------------------------------------
    def _reset_episode_state(self):
        """Reset all per-episode state before a new episode."""
        self.episode_done.clear()
        self.must_go.set()
        self.latest_action = -1
        self.action_chunk_size = -1
        self.action_queue = Queue()
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(3)
        # Convergence tracking
        self._action_history = []
        self._first_action = None
        self._max_displacement = 0.0
        self._converged = False
        self._total_actions = 0
        self._last_action_time = 0.0

    # ------------------------------------------------------------------
    # Episode threads
    # ------------------------------------------------------------------
    def _receive_actions_thread(self):
        """Background thread: receives action chunks from the policy server."""
        self.start_barrier.wait()
        self.logger.info("Action receiver thread started")

        while not self.episode_done.is_set():
            try:
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue

                timed_actions = pickle.loads(actions_chunk.data)  # nosec

                # Move to client device if needed
                client_device = self.config.client_device
                if client_device != "cpu":
                    for ta in timed_actions:
                        if ta.get_action().device.type != client_device:
                            ta.action = ta.get_action().to(client_device)

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                self.must_go.set()

            except grpc.RpcError as e:
                if not self.episode_done.is_set():
                    self.logger.error(f"Error receiving actions: {e}")

    def _abort_listener_thread(self):
        """Listen for Enter key to abort episode early."""
        import select
        import sys
        self.start_barrier.wait()
        while not self.episode_done.is_set():
            # Poll stdin with 0.1s timeout so we can check episode_done
            if select.select([sys.stdin], [], [], 0.1)[0]:
                sys.stdin.readline()
                if not self.episode_done.is_set():
                    self.logger.info("Episode aborted by user.")
                    self.episode_done.set()
                break

    def _control_loop_thread(self, task: str):
        """Main thread: sends observations and executes actions until episode ends."""
        self.start_barrier.wait()
        self.logger.info(f"Control loop started | task: '{task}' | duration: {self.config.episode_duration}s")

        episode_start = time.perf_counter()
        self._episode_start_perf = episode_start

        while not self.episode_done.is_set():
            loop_start = time.perf_counter()
            elapsed = loop_start - episode_start

            # Check timeout
            if elapsed >= self.config.episode_duration:
                self.logger.info(f"Episode timeout ({self.config.episode_duration}s)")
                self.episode_done.set()
                break

            # Execute action if available
            with self.action_queue_lock:
                has_action = not self.action_queue.empty()

            if has_action:
                try:
                    with self.action_queue_lock:
                        self.action_queue_size.append(self.action_queue.qsize())
                        timed_action = self.action_queue.get_nowait()

                    action_tensor = timed_action.get_action()
                    self.robot.send_action(
                        self._action_tensor_to_action_dict(action_tensor)
                    )

                    with self.latest_action_lock:
                        self.latest_action = timed_action.get_timestep()

                    # --- Convergence tracking ---
                    action_np = action_tensor.cpu().numpy().flatten()
                    self._total_actions += 1
                    self._last_action_time = elapsed

                    if self._first_action is None:
                        self._first_action = action_np.copy()

                    # Track max displacement from first action
                    displacement = float(np.linalg.norm(action_np - self._first_action))
                    if displacement > self._max_displacement:
                        self._max_displacement = displacement

                    # Rolling window for convergence check
                    self._action_history.append(action_np)
                    if len(self._action_history) > self.config.convergence_window:
                        self._action_history.pop(0)

                    # Check convergence after grace period
                    if (elapsed >= self.config.convergence_grace_period
                            and len(self._action_history) == self.config.convergence_window):
                        # Max pairwise L2 between consecutive actions in the window
                        max_delta = 0.0
                        for j in range(1, len(self._action_history)):
                            delta = float(np.max(np.abs(
                            self._action_history[j] - self._action_history[j - 1]
                        )))
                            if delta > max_delta:
                                max_delta = delta

                        if max_delta < self.config.convergence_threshold:
                            self._converged = True
                            self.logger.info(
                                f"Action convergence detected at {elapsed:.1f}s "
                                f"(max_delta={max_delta:.4f}, "
                                f"threshold={self.config.convergence_threshold}). "
                                f"Max displacement from start: {self._max_displacement:.4f}"
                            )
                            self.episode_done.set()
                            break
                        else:
                            self.logger.info(
                                f"Convergence check at {elapsed:.1f}s: "
                                f"max_delta={max_delta:.6f} "
                                f"(threshold={self.config.convergence_threshold})"
                            )

                except Empty:
                    pass

            else:
                # --- Stale-action detection ---
                # If the policy server stops sending actions (queue empty) but
                # we've already received some, the arm has effectively stopped.
                # This catches the case where π0.5 stops producing chunks when
                # it considers the task done, so the in-window convergence check
                # never fires because no new actions enter the buffer.
                if (self._total_actions > 0
                        and elapsed >= self.config.convergence_grace_period):
                    time_since_last = elapsed - self._last_action_time
                    if time_since_last >= self.config.action_stall_timeout:
                        self._converged = True
                        self.logger.info(
                            f"Action stall detected at {elapsed:.1f}s — no new "
                            f"actions for {time_since_last:.1f}s. "
                            f"Max displacement from start: {self._max_displacement:.4f}"
                        )
                        self.episode_done.set()
                        break

            # Send observation if ready
            if self._ready_to_send_observation():
                try:
                    raw_observation: RawObservation = self.robot.get_observation()
                    raw_observation["task"] = task

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    observation = TimedObservation(
                        timestamp=time.time(),
                        observation=raw_observation,
                        timestep=max(latest_action, 0),
                    )

                    with self.action_queue_lock:
                        observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                    self.send_observation(observation)
                    if observation.must_go:
                        self.must_go.clear()
                except Exception as e:
                    self.logger.error(f"Error in observation sender: {e}")

            # Maintain control frequency
            time.sleep(
                max(0, self.config.environment_dt - (time.perf_counter() - loop_start))
            )

    # ------------------------------------------------------------------
    # Run a single episode
    # ------------------------------------------------------------------
    def run_episode(self, task: str) -> EpisodeResult:
        """Execute one episode: run the policy for the given task until timeout or convergence."""
        self._reset_episode_state()

        receiver = threading.Thread(target=self._receive_actions_thread, daemon=True)
        receiver.start()

        abort = threading.Thread(target=self._abort_listener_thread, daemon=True)
        abort.start()

        self.logger.info("Press Enter to abort episode early.")
        self._control_loop_thread(task)  # blocks until episode_done
        receiver.join(timeout=5.0)
        abort.join(timeout=1.0)

        result = EpisodeResult(
            duration=time.perf_counter() - self._episode_start_perf,
            converged=self._converged,
            max_displacement=self._max_displacement,
            total_actions=self._total_actions,
        )
        self.logger.info(
            f"Episode finished: {result.duration:.1f}s, converged={result.converged}, "
            f"max_displacement={result.max_displacement:.4f}, "
            f"actions={result.total_actions}"
        )
        return result

    # ------------------------------------------------------------------
    # Capture a frame (for VLM evaluation later)
    # ------------------------------------------------------------------
    def capture_frame(self) -> dict:
        """Capture a single observation frame (useful for VLM success evaluation)."""
        return self.robot.get_observation()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def stop(self):
        self.shutdown_event.set()
        self.episode_done.set()
        self.robot.disconnect()
        self.channel.close()
        self.logger.info("Client stopped")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
@draccus.wrap()
def main(cfg: LoopClientConfig):
    logging.basicConfig(level=logging.INFO)
    logging.info(pformat(asdict(cfg)))

    client = LoopRobotClient(cfg)

    if not client.connect_server():
        client.logger.error("Could not connect to policy server. Exiting.")
        client.stop()
        return

    client.logger.info("=" * 60)
    client.logger.info("Policy server connected. Model loaded and ready.")
    client.logger.info(f"Episode duration: {cfg.episode_duration}s")
    client.logger.info("Type a task prompt and press Enter to execute.")
    client.logger.info("Type 'quit' or 'exit' to shut down.")
    client.logger.info("=" * 60)

    try:
        while True:
            # Wait for task input
            try:
                task = input("\n[READY] Enter task prompt: ").strip()
            except EOFError:
                break

            if not task:
                continue
            if task.lower() in ("quit", "exit"):
                client.logger.info("Shutdown requested.")
                break

            # Run the episode
            client.logger.info(f"Starting episode: '{task}'")
            result = client.run_episode(task)

            # Capture final frame (available for VLM evaluation later)
            final_obs = client.capture_frame()
            client.logger.info(
                f"Episode complete. Final observation keys: {list(final_obs.keys())}"
            )

            # Return to home
            client.go_home()

    except KeyboardInterrupt:
        client.logger.info("\nInterrupted by user.")

    finally:
        client.go_home()
        client.stop()


if __name__ == "__main__":
    register_third_party_plugins()
    main()