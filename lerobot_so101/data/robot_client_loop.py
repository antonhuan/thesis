# robot_client_loop.py
#
# Modified from lerobot/async_inference/robot_client.py
# Adds a persistent task loop: the policy server connection and model stay
# loaded, and between episodes the arm returns to its home position and
# waits for the next task prompt via stdin.
#
# Usage:
#   Terminal 1 (policy server, stays running):
#     python fast_policy_server.py --host=127.0.0.1 --port=8080
#     (same flags as `python -m lerobot.async_inference.policy_server`, but skips
#      the ~3 min random init pi05 otherwise pays for on every load)
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
import re
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
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
# Fixed home/rest pose. The arm is returned here between episodes instead of to
# whatever pose it happened to be in at startup, so homing is reproducible
# across runs. Keys are joint names; the ".pos" suffix is matched loosely
# against whatever the robot's action features are called.
# ---------------------------------------------------------------------------
ACTION_LOG_FORMATTER = logging.Formatter(
    "%(levelname)s %(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)


HOME_POSITION: dict[str, float] = {
    "shoulder_pan.pos": -4.571428571428571,
    "shoulder_lift.pos": -101.49450549450549,
    "elbow_flex.pos": 91.91208791208791,
    "wrist_flex.pos": 74.28571428571429,
    "wrist_roll.pos": -0.7472527472527473,
    "gripper.pos": 1.3013698630136987,
}


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
    # Fixed joint targets used as the home/rest pose (see HOME_POSITION above)
    home_position: dict[str, float] = field(default_factory=lambda: dict(HOME_POSITION))
    # Number of consecutive similar actions before declaring convergence
    convergence_window: int = 25
    # L2 threshold: actions closer than this are "similar"
    convergence_threshold: float = 1
    # Don't check convergence in the first N seconds (let the arm start moving)
    convergence_grace_period: float = 6
    # Seconds with no new actions before declaring a stall (policy stopped sending)
    action_stall_timeout: float = 2.0
    # Buffer per-episode camera frames so the VLM can be shown a short video clip
    # of the attempt (used by the evaluation step).
    enable_clip_buffer: bool = True
    # Max frames retained in the episode clip buffer. The buffer decimates (drops
    # every other frame) rather than evicting its head when it fills, so it always
    # spans the whole episode start->now; this bounds memory while keeping full
    # temporal coverage. Higher = finer retained resolution before subsampling.
    clip_buffer_maxlen: int = 128
    # Log the joint targets actually sent to the arm. Actions arrive at ~fps, so
    # they go to their own file (see action_log_dir) instead of the main client
    # log, which they would otherwise bury. Nothing action-related is ever
    # written to the terminal: tail the action log to watch a run live.
    log_actions: bool = True
    # Directory for the per-run action log; the file is named
    # <prefix>_actions_<unix_ts>.log to pair with the main client log.
    action_log_dir: str = "logs"
    # Also log each action chunk as it arrives from the policy server (size and
    # timestep range) to the action log.
    log_action_chunks: bool = True
    # Capture each VLA forward-pass input (scene image + joint state + task) paired
    # with the actions that observation's chunk actually put on the arm (the
    # executed subset), for prompt/error analysis.
    log_vla_inputs: bool = True
    # Root directory for VLA-input logging. Each episode gets its own
    # <index>_<object>/ subdirectory here, holding that episode's input PNGs, an
    # actions.csv (one row per executed action, paired with its input) and a
    # recap.png contact sheet for manual outcome labelling.
    vla_input_dir: str = "vla_inputs"
    # Camera key used for both the VLA-input snapshot and the episode clip buffer
    # (see _select_camera_frame). Must match the scene camera the policy is
    # trained on (front-facing), not an arm-mounted camera like wrist.
    vlm_camera_key: str = "front"


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
      2. Uses a fixed, configured pose as "home" (see HOME_POSITION)
      3. Waits for a task prompt from stdin
      4. Runs the async control loop for --episode_duration seconds
      5. Returns the arm to home
      6. Goes back to step 3
    """

    prefix = "loop_client"
    logger = get_logger(prefix)

    def __init__(self, config: LoopClientConfig):
        self.config = config
        self.action_log_path: Path | None = None
        self.action_logger = self._setup_action_logger()
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

        # Episode clip buffer: per-episode (timestamp, frame) pairs so the VLM can
        # be shown a short video of the attempt. Timestamps drive even-in-time
        # subsampling. The buffer uniformly downsamples the frame stream when full
        # (see _buffer_clip_frame) so it spans the whole episode at even density
        # rather than retaining only its tail; the clip's real time span still
        # determines its frame rate.
        self._clip_camera_key = config.vlm_camera_key
        self._episode_clip: list = []
        # Keep every _clip_stride-th candidate frame; doubles each time the buffer
        # fills so retained frames stay evenly spaced across the whole episode.
        self._clip_stride = 1
        self._clip_seen = 0
        self._clip_lock = threading.Lock()

        # VLA-input capture: save the scene image + joint state + task for each
        # forward-pass observation, paired with the actions that observation's
        # chunk actually put on the arm. Each episode gets its own subdirectory
        # (see _start_vla_episode_dir) with its own image sequence, an actions.csv
        # (one row per executed action), and a recap.png for manual labelling.
        #
        # Inputs are captured for every observation the client sends, keyed by the
        # client-assigned obs timestep (see _capture_input). The policy server
        # stamps each chunk's first action with that same obs timestep
        # (policy_server._time_action_chunk), so a received chunk binds to its
        # input by matching timestep (_bind_chunk_input) — no mis-pairing, and
        # stale cross-episode chunks are dropped. Every action actually sent to the
        # arm is attributed back to the chunk that supplied it via a timestep->
        # chunk_id source map (_timestep_source), so only the executed subset of
        # each chunk is written (_flush_executed_chunks).
        self._vla_input_root = Path(config.vla_input_dir)
        self._vla_input_dir = self._vla_input_root
        self._vla_input_count = 0
        self._vla_episode_index = 0
        self._vla_lock = threading.Lock()
        self._sent_inputs: dict[int, dict] = {}     # obs.timestep -> pending input record
        self._active_chunks: dict[int, dict] = {}   # chunk_id -> bound input + executed lists
        self._timestep_source: dict[int, int] = {}  # action timestep -> owning chunk_id
        self._chunk_counter = 0
        # Bound memory for inputs whose observation the server never predicts.
        self._sent_inputs_maxlen = 64
        if config.log_vla_inputs:
            self._vla_input_root.mkdir(parents=True, exist_ok=True)
            # Resume the episode index from whatever's already on disk. It would
            # otherwise restart at 0 on every process launch, and the first
            # episode of a new run would silently reuse (and overwrite into) the
            # first episode's directory from a *previous* run.
            self._vla_episode_index = self._next_vla_episode_index()
            self.logger.info(
                f"Logging VLA inputs under {self._vla_input_root}/, one subdirectory "
                f"per episode (starting at index {self._vla_episode_index})"
            )

        # Convergence tracking (initialised properly in _reset_episode_state)
        self._action_history = []
        self._first_action = None
        self._max_displacement = 0.0
        self._converged = False
        self._total_actions = 0
        self._last_action_time = 0.0
        self._last_action_dict = None

        # Fixed home position (not whatever pose the arm starts in)
        self.home_position = self._resolve_home_position(config.home_position)
        self.logger.info(f"Home position (fixed): {self.home_position}")

        self.logger.info("Robot connected and ready")

    # ------------------------------------------------------------------
    # Action logging
    # ------------------------------------------------------------------
    def _setup_action_logger(self) -> logging.Logger:
        """Build the dedicated logger for per-action output.

        Actions arrive at ~fps, so they get their own file rather than sharing
        the main client log. The logger has no console handler and does not
        propagate to the root logger, so actions never reach the terminal (and,
        in the orchestrator, are not picked up by the stdout/stderr tee either):
        the action log file is the only sink.
        """
        logger = logging.getLogger(f"{self.prefix}_actions")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        # A second client in the same process would otherwise double up handlers
        # (and keep writing to the previous run's file).
        self._clear_action_handlers(logger)

        if not self.config.log_actions:
            logger.addHandler(logging.NullHandler())
            return logger

        log_dir = Path(self.config.action_log_dir)
        default_path = log_dir / f"{self.prefix}_actions_{int(time.time())}.log"
        self._attach_action_file_handler(logger, default_path)
        self.logger.info(f"Action log: {self.action_log_path}")
        return logger

    @staticmethod
    def _clear_action_handlers(logger: logging.Logger) -> None:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    def _attach_action_file_handler(self, logger: logging.Logger, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # delay=True: no file is created until an action is actually logged, so
        # repointing the log per task leaves no empty stragglers behind.
        file_handler = logging.FileHandler(path, delay=True)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(ACTION_LOG_FORMATTER)
        logger.addHandler(file_handler)
        self.action_log_path = path

    def start_action_log(self, path: Path | str) -> Path | None:
        """Point the action log at `path`, closing the previous file.

        Lets a caller give each task its own action log (the orchestrator puts
        one next to that task's task.log). Returns the new path, or None when
        action logging is disabled.
        """
        if not self.config.log_actions:
            return None
        self._clear_action_handlers(self.action_logger)
        self._attach_action_file_handler(self.action_logger, Path(path))
        return self.action_log_path

    # ------------------------------------------------------------------
    # Home position helpers
    # ------------------------------------------------------------------
    def _resolve_home_position(self, home: dict[str, float]) -> dict[str, float]:
        """Map the configured home pose onto the robot's action feature keys.

        The configured keys carry a ".pos" suffix; the robot's action features
        may or may not, so match both ways and fall back to the arm's current
        reading for any joint the config doesn't mention.
        """
        current = self._capture_current_position()
        resolved = {}
        for key in self.robot.action_features:
            if key in home:
                resolved[key] = float(home[key])
            elif f"{key}.pos" in home:
                resolved[key] = float(home[f"{key}.pos"])
            elif key.removesuffix(".pos") in home:
                resolved[key] = float(home[key.removesuffix(".pos")])
            else:
                resolved[key] = current.get(key, 0.0)
                self.logger.warning(
                    f"No home position configured for {key}; using current value {resolved[key]}"
                )
        return resolved

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
        chunk_id: int | None = None,
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

        # The new queue is rebuilt from this chunk's actions, so every timestep it
        # ends up owning is sourced to chunk_id (latest-wins, matching the default
        # aggregate_fn) — this is what later attributes each executed action to the
        # chunk that produced it (see _attribute_executed).
        queued_timesteps: list[int] = []
        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            ts = new_action.get_timestep()
            if ts <= latest_action:
                continue
            elif ts not in current_action_queue:
                future_action_queue.put(new_action)
            else:
                future_action_queue.put(
                    TimedAction(
                        timestamp=new_action.get_timestamp(),
                        timestep=ts,
                        action=aggregate_fn(
                            current_action_queue[ts],
                            new_action.get_action(),
                        ),
                    )
                )
            queued_timesteps.append(ts)

        with self.action_queue_lock:
            self.action_queue = future_action_queue

        if chunk_id is not None and queued_timesteps:
            with self._vla_lock:
                for ts in queued_timesteps:
                    self._timestep_source[ts] = chunk_id

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        return {
            key: action_tensor[i].item()
            for i, key in enumerate(self.robot.action_features)
        }

    @staticmethod
    def _format_action(action: dict[str, float]) -> str:
        """Compact one-line rendering of a joint-target dict for the logs."""
        return " ".join(f"{key}={value:+.2f}" for key, value in action.items())

    def _log_action(self, action: dict[str, float], timestep: int, elapsed: float,
                    queue_size: int, delta: float | None):
        """Log a single action sent to the arm to the dedicated action log.

        Every action is written to the action log file and nowhere else.
        `delta` is the max per-joint change from the previous action, or None
        for the first action of the episode.
        """
        if not self.config.log_actions:
            return

        delta_str = f"{delta:.3f}" if delta is not None else "n/a"
        self.action_logger.debug(
            f"[action #{self._total_actions}] t={elapsed:.2f}s step={timestep} "
            f"queue={queue_size} dmax={delta_str} | {self._format_action(action)}"
        )

    def _ready_to_send_observation(self) -> bool:
        with self.action_queue_lock:
            return self.action_queue.qsize() / max(self.action_chunk_size, 1) <= self._chunk_size_threshold

    # ------------------------------------------------------------------
    # Episode state management
    # ------------------------------------------------------------------
    def _reset_episode_state(self, task: str):
        """Reset all per-episode state before a new episode."""
        if self.config.log_vla_inputs:
            # Defensive: run_episode flushes at episode end, but never carry stale
            # capture state into the new episode's directory.
            with self._vla_lock:
                self._sent_inputs = {}
                self._active_chunks = {}
                self._timestep_source = {}
                self._chunk_counter = 0
            self._start_vla_episode_dir(task)
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
        self._last_action_dict = None
        # Start each episode's clip fresh
        with self._clip_lock:
            self._episode_clip.clear()
            self._clip_stride = 1
            self._clip_seen = 0

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

                if self.config.log_actions and self.config.log_action_chunks:
                    timesteps = [ta.get_timestep() for ta in timed_actions]
                    first = self._action_tensor_to_action_dict(timed_actions[0].get_action())
                    last = self._action_tensor_to_action_dict(timed_actions[-1].get_action())
                    self.action_logger.debug(
                        f"[chunk] received {len(timed_actions)} actions, "
                        f"timesteps {min(timesteps)}..{max(timesteps)} | "
                        f"first: {self._format_action(first)} | "
                        f"last: {self._format_action(last)}"
                    )

                chunk_id = None
                if self.config.log_vla_inputs and timed_actions:
                    with self._vla_lock:
                        self._chunk_counter += 1
                        chunk_id = self._chunk_counter
                    # Bind this chunk to the input that produced it, by matching
                    # the chunk's first action timestep (== the observation's
                    # timestep) to a captured input.
                    self._bind_chunk_input(
                        chunk_id, timed_actions[0].get_timestep(), len(timed_actions)
                    )

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))
                self._aggregate_action_queues(
                    timed_actions, self.config.aggregate_fn, chunk_id=chunk_id
                )
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
        # Episode boundary marker in the action log (file only), so the action
        # stream can be split by task without cross-referencing the main log.
        self.action_logger.debug(
            f"=== episode start | task: '{task}' | "
            f"duration: {self.config.episode_duration}s ==="
        )

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
                    action_dict = self._action_tensor_to_action_dict(action_tensor)
                    self.robot.send_action(action_dict)

                    # Attribute this executed action to the chunk (hence input)
                    # that supplied its value, for the per-input executed subset.
                    # action_n is the per-episode action number and must match the
                    # action log's [action #N], which logs the post-increment value
                    # (self._total_actions is bumped below), so pass +1 here.
                    if self.config.log_vla_inputs:
                        self._attribute_executed(
                            timed_action.get_timestep(), action_dict,
                            action_n=self._total_actions + 1,
                        )

                    with self.latest_action_lock:
                        self.latest_action = timed_action.get_timestep()

                    # --- Convergence tracking ---
                    action_np = action_tensor.cpu().numpy().flatten()
                    self._total_actions += 1
                    self._last_action_time = elapsed

                    # --- Action logging ---
                    previous = self._action_history[-1] if self._action_history else None
                    step_delta = (
                        float(np.max(np.abs(action_np - previous)))
                        if previous is not None
                        else None
                    )
                    self._last_action_dict = action_dict
                    self._log_action(
                        action_dict,
                        timestep=timed_action.get_timestep(),
                        elapsed=elapsed,
                        queue_size=self.action_queue_size[-1],
                        delta=step_delta,
                    )

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
                            self.logger.debug(
                                f"Action convergence detected at {elapsed:.1f}s "
                                f"(max_delta={max_delta:.4f}, "
                                f"threshold={self.config.convergence_threshold}). "
                                f"Max displacement from start: {self._max_displacement:.4f}"
                            )
                            self.episode_done.set()
                            break
                        else:
                            self.logger.debug(
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

                    # Buffer this frame for the episode clip (VLM video input)
                    if self.config.enable_clip_buffer:
                        self._buffer_clip_frame(raw_observation)

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
                    # Capture the VLA input for every observation sent (not only
                    # must_go): the server predicts any observation that clears its
                    # gate, so the chunk that binds to this input may come from a
                    # non-must_go send. The input is held in memory and only
                    # written if a chunk binds to it and executes (_capture_input /
                    # _bind_chunk_input / _flush_executed_chunks).
                    self._capture_input(
                        raw_observation, task,
                        observation.get_timestep(), elapsed,
                        must_go=observation.must_go,
                    )
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
    def run_episode(self, task: str, enable_abort_listener: bool = True) -> EpisodeResult:
        """Execute one episode: run the policy for the given task until timeout or convergence."""
        self._reset_episode_state(task)

        if not enable_abort_listener:
            # When an external manager handles stdin, skip the abort thread
            # and reduce the barrier from 3 to 2 parties.
            self.start_barrier = threading.Barrier(2)

        receiver = threading.Thread(target=self._receive_actions_thread, daemon=True)
        receiver.start()

        abort = None
        if enable_abort_listener:
            abort = threading.Thread(target=self._abort_listener_thread, daemon=True)
            abort.start()
            self.logger.info("Press Enter to abort episode early.")

        self._control_loop_thread(task)  # blocks until episode_done
        receiver.join(timeout=5.0)
        if abort is not None:
            abort.join(timeout=1.0)

        # Receiver and control threads have stopped: no more chunks will bind and
        # no more actions will execute. Write the executed subset of each bound
        # chunk (paired with its input) to actions.csv and render the recap.
        if self.config.log_vla_inputs:
            self._flush_executed_chunks()

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
        if self.config.log_actions and self._last_action_dict is not None:
            self.action_logger.info(
                f"Final action sent: {self._format_action(self._last_action_dict)}"
            )
            self.action_logger.debug(
                f"=== episode end | {result.duration:.1f}s | "
                f"converged={result.converged} | actions={result.total_actions} ==="
            )
        return result

    # ------------------------------------------------------------------
    # Capture a frame (for VLM evaluation later)
    # ------------------------------------------------------------------
    def capture_frame(self) -> dict:
        """Capture a single observation frame (useful for VLM success evaluation)."""
        return self.robot.get_observation()

    # ------------------------------------------------------------------
    # Episode clip buffer (for VLM video input)
    # ------------------------------------------------------------------
    @staticmethod
    def _select_camera_frame(obs: dict, camera_key) -> "np.ndarray | None":
        """Pick an HxWx3 uint8-able array from an observation dict.

        Prefers camera_key; falls back to the first image-like array. Mirrors the
        selection logic in the orchestrator's VLMFrameSource.capture.
        """
        value = obs.get(camera_key) if camera_key is not None else None
        if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] == 3:
            return value
        for v in obs.values():
            if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] == 3:
                return v
        return None

    @staticmethod
    def _select_all_camera_frames(obs: dict) -> dict:
        """Return every HxWx3 uint8-able array in obs, keyed by its observation key.

        Used for VLA-input logging: the policy is fed every camera view (see
        pi05_client.py, which sends both images/front and images/wrist), so the
        saved record should capture all of them, not just one.
        """
        return {
            key: value for key, value in obs.items()
            if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] == 3
        }

    def _buffer_clip_frame(self, obs: dict) -> None:
        """Append a timestamped copy of the selected camera frame to the clip buffer.

        Uniformly downsamples the frame stream so the bounded buffer spans the whole
        episode at even density instead of retaining only its tail: only every
        _clip_stride-th candidate is kept, and when the buffer fills it is halved
        (keep every other retained frame) and the stride doubled. Incoming frames
        then arrive at the same effective interval as the halved older ones, so the
        first frame (episode start) through the newest stay evenly spaced. Memory is
        bounded by clip_buffer_maxlen.
        """
        frame = self._select_camera_frame(obs, self._clip_camera_key)
        if frame is None:
            return
        with self._clip_lock:
            self._clip_seen += 1
            if self._clip_seen % self._clip_stride != 0:
                return
            self._episode_clip.append((time.perf_counter(), frame.copy()))
            if len(self._episode_clip) > self.config.clip_buffer_maxlen:
                # Halve the buffer (keep indices 0, 2, 4, ... — retains the
                # episode-start frame) and match the incoming rate to it.
                self._episode_clip = self._episode_clip[::2]
                self._clip_stride *= 2

    @staticmethod
    def _slugify_task(task: str) -> str:
        """Filesystem-safe, length-capped rendering of a task string."""
        return "".join(c if c.isalnum() else "_" for c in task).strip("_")[:40]

    @staticmethod
    def _extract_object(task: str) -> str:
        """Best-effort object tag for episode-directory naming: the first word
        following "the" in the task string (e.g. "put the banana on the tray"
        -> "banana"). Falls back to the full slugified task if "the" doesn't
        appear, so every episode still gets a usable, unique directory name.
        Deliberately simple/generic rather than matched against a fixed
        vocabulary, so it degrades gracefully on tasks outside that vocabulary
        too (see vla_denoise_consistency.py's replay-mode group resolution,
        which has its own fallback for when this heuristic misses).
        """
        match = re.search(r"\bthe\s+(\w+)", task, re.IGNORECASE)
        return match.group(1).lower() if match else LoopRobotClient._slugify_task(task)

    def _next_vla_episode_index(self) -> int:
        """Pick the next episode index by scanning vla_input_root for existing
        <index>_<object>/ directories (at any depth, since episodes are nested
        under an <object>/ subdirectory -- see _start_vla_episode_dir), so a
        fresh process resumes numbering after whatever a previous run already
        wrote instead of restarting at 0 and colliding with (and overwriting
        into) that run's first episode dir. Recursive rather than one level
        deep so it also picks up any leftover flat-layout episode dirs from
        before object-subdirectory nesting was added.
        """
        max_idx = -1
        for entry in self._vla_input_root.rglob("*"):
            if not entry.is_dir():
                continue
            prefix = entry.name.split("_", 1)[0]
            if prefix.isdigit():
                max_idx = max(max_idx, int(prefix))
        return max_idx + 1

    def _start_vla_episode_dir(self, task: str) -> None:
        """Point VLA-input logging at a fresh subdirectory for this episode.

        Each episode gets its own <object>/<index>_<object>/ directory under
        the configured vla_input_dir -- grouped by object so a downstream
        script can sweep every episode for one object in one pass -- with its
        own image sequence starting back at 00000, its own actions.csv, and its
        own recap.png. This keeps one episode's inputs/outputs self-contained.
        The directory name uses just the object tag (not the full task text) so
        it can be parsed back out directly.
        """
        object_tag = self._extract_object(task)
        ep_dir = (
            self._vla_input_root / object_tag
            / f"{self._vla_episode_index:03d}_{object_tag}"
        )
        ep_dir.mkdir(parents=True, exist_ok=True)
        self._vla_input_dir = ep_dir
        self._vla_input_count = 0
        self._vla_episode_index += 1
        self.logger.info(f"VLA inputs for this episode -> {ep_dir}/")

    def _capture_input(self, obs: dict, task: str, timestep: int, elapsed: float,
                       must_go: bool) -> None:
        """Stash the VLA forward-pass input for one sent observation.

        Called for every observation the client sends. The camera frames + joint
        state are held in memory keyed by the client-assigned obs timestep; a
        later chunk whose first action timestep matches this key binds to the
        record (_bind_chunk_input) and only then are the frames written to disk.
        Observations the server never predicts never bind and are pruned, so
        nothing is saved for them. On a same-timestep collision the earliest
        capture is kept (the server predicts the first observation it enqueues for
        a timestep), except a must_go capture replaces a non-must_go one.
        """
        if not self.config.log_vla_inputs:
            return

        frames = self._select_all_camera_frames(obs)
        if not frames:
            return

        # Joint state in motor order, mirroring _capture_current_position's key
        # handling (the action_features order is also the CSV joint-column order).
        state = []
        for key in self.robot.action_features:
            obs_key = f"{key}.pos" if f"{key}.pos" in obs else key
            state.append(float(obs[obs_key]) if obs_key in obs else 0.0)

        record = {
            # Copies: the raw obs arrays are reused/overwritten by the camera loop.
            "frames": {k: np.asarray(v, dtype=np.uint8).copy() for k, v in frames.items()},
            "task": task,
            "state": state,
            "timestep": int(timestep),
            "elapsed": round(float(elapsed), 3),
            "unix_ts": time.time(),
            "must_go": bool(must_go),
        }
        with self._vla_lock:
            existing = self._sent_inputs.get(int(timestep))
            if existing is None or (must_go and not existing.get("must_go")):
                self._sent_inputs[int(timestep)] = record
            # Bound memory: drop oldest pending inputs never bound to a chunk.
            if len(self._sent_inputs) > self._sent_inputs_maxlen:
                for old in sorted(self._sent_inputs)[:-self._sent_inputs_maxlen]:
                    del self._sent_inputs[old]

    def _bind_chunk_input(self, chunk_id: int, ts0: int, returned_len: int) -> None:
        """Bind a received chunk to the input observation that produced it.

        The policy server stamps the chunk's first action with the observation's
        own timestep (policy_server._time_action_chunk), so ts0 identifies the
        exact input the client sent. A chunk whose ts0 matches no pending input (a
        stale cross-episode chunk still in the server's queue) is left unbound and
        contributes no rows.
        """
        with self._vla_lock:
            record = self._sent_inputs.pop(int(ts0), None)
            if record is None:
                return
            record["chunk_id"] = chunk_id
            record["returned_len"] = int(returned_len)
            record["executed_actions"] = []
            record["executed_timesteps"] = []
            record["executed_action_ns"] = []
            self._active_chunks[chunk_id] = record

    def _attribute_executed(self, timestep: int, action_dict: dict, action_n: int) -> None:
        """Attribute one executed action to the chunk (hence input) that produced it.

        The value at `timestep` in the action queue was written by whichever chunk
        the source map last recorded (_aggregate_action_queues), so the executed
        action belongs to that chunk's input. `action_n` is the per-episode action
        number (matches [action #N] in the action log); it is the reference used
        downstream to locate a moment within an episode, since the server timestep
        carries over across episodes.
        """
        with self._vla_lock:
            chunk_id = self._timestep_source.get(int(timestep))
            record = self._active_chunks.get(chunk_id) if chunk_id is not None else None
            if record is None:
                return
            record["executed_actions"].append(dict(action_dict))
            record["executed_timesteps"].append(int(timestep))
            record["executed_action_ns"].append(int(action_n))

    def _flush_executed_chunks(self) -> None:
        """Write each bound chunk's executed subset (+ its input) to actions.csv.

        Called at episode end. Each chunk that put at least one action on the arm
        contributes its executed actions (in execution order) as CSV rows, paired
        with the input that produced it; its held frames are saved as PNGs. Chunks
        that never executed (fully superseded before reaching the arm) and inputs
        that never bound are dropped. Also renders the per-episode recap image.
        """
        if not self.config.log_vla_inputs:
            return
        with self._vla_lock:
            chunks = [self._active_chunks[cid] for cid in sorted(self._active_chunks)]
            self._active_chunks = {}
            self._sent_inputs = {}
            self._timestep_source = {}
            self._chunk_counter = 0

        rows = [rec for rec in chunks if rec.get("executed_actions")]
        if not rows:
            return
        for rec in rows:
            rec["images"] = self._save_input_frames(rec)
        recap_name = self._render_episode_recap(rows)
        self._write_actions_csv(rows, recap_name)

    def _save_input_frames(self, rec: dict) -> dict:
        """Persist a bound input's held camera frames as PNGs; return {cam: fname}."""
        from PIL import Image

        safe_task = self._slugify_task(rec["task"])
        images: dict[str, str] = {}
        for cam_key, frame in rec.get("frames", {}).items():
            fname = f"{self._vla_input_count:05d}_{cam_key}_{safe_task}.png"
            try:
                Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(
                    self._vla_input_dir / fname
                )
            except Exception as e:  # noqa: BLE001 - never let logging break an episode
                self.logger.error(f"Failed to save VLA input image {fname}: {e}")
                continue
            images[cam_key] = fname
        self._vla_input_count += 1
        return images

    def _write_actions_csv(self, records: list, recap_name: str | None) -> None:
        """Append the executed actions of each record to the episode's actions.csv.

        Long format: one row per action actually sent to the arm, so the nested
        chunks flatten and the file drops straight into pandas. Rows for one
        inference share chunk_id / input_timestep / images; sum of rows for the
        episode equals its executed-action count.
        """
        import csv

        joints = list(self.robot.action_features)
        header = (
            ["episode_dir", "task", "chunk_id", "must_go", "input_timestep",
             "input_elapsed", "unix_ts", "front_img", "wrist_img", "recap_img",
             "returned_len", "exec_index", "exec_timestep", "exec_action_n"]
            + [f"in_{j}" for j in joints]
            + [f"act_{j}" for j in joints]
        )
        path = self._vla_input_dir / "actions.csv"
        write_header = not path.exists()
        episode_dir = self._vla_input_dir.name
        try:
            with open(path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(header)
                for rec in records:
                    imgs = rec.get("images", {})
                    state = rec.get("state", [])
                    in_vals = [state[i] if i < len(state) else "" for i in range(len(joints))]
                    for idx, (act, ts, act_n) in enumerate(
                        zip(rec["executed_actions"], rec["executed_timesteps"],
                            rec["executed_action_ns"])
                    ):
                        writer.writerow(
                            [episode_dir, rec["task"], rec["chunk_id"],
                             int(rec["must_go"]), rec["timestep"], rec["elapsed"],
                             rec["unix_ts"], imgs.get("front", ""), imgs.get("wrist", ""),
                             recap_name or "", rec.get("returned_len", ""), idx, ts, act_n]
                            + in_vals
                            + [act.get(j, "") for j in joints]
                        )
        except Exception as e:  # noqa: BLE001
            self.logger.error(f"Failed to write actions.csv: {e}")

    def _render_episode_recap(self, records: list, num_frames: int = 16) -> str | None:
        """Render an annotated contact sheet of the episode for manual labelling.

        Frames are sampled evenly in time from the clip buffer; each tile is
        labelled with its elapsed time and the per-episode action number of the
        inference active then (nearest by elapsed), so a tile lines up with the
        actions.csv rows (exec_action_n) and the action log ([action #N]).
        Returns the saved filename, or None if nothing was buffered.
        """
        if not self.config.enable_clip_buffer:
            return None
        with self._clip_lock:
            entries = list(self._episode_clip)
        if not entries:
            return None

        t0 = entries[0][0]
        span = entries[-1][0] - t0 if len(entries) > 1 else 0.0
        if len(entries) <= num_frames or span <= 0:
            sampled = entries
        else:
            times = [ts for ts, _ in entries]
            idxs: list[int] = []
            for k in range(num_frames):
                target = t0 + span * k / (num_frames - 1)
                nearest = min(range(len(entries)), key=lambda i: abs(times[i] - target))
                if nearest not in idxs:
                    idxs.append(nearest)
            idxs.sort()
            sampled = [entries[i] for i in idxs]

        episode_start = getattr(self, "_episode_start_perf", t0)
        try:
            from PIL import Image, ImageDraw

            tiles = []
            for ts, frame in sampled:
                img = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
                elapsed = ts - episode_start
                label = f"t={elapsed:4.1f}s"
                if records:
                    nearest = min(records, key=lambda r: abs(r["elapsed"] - elapsed))
                    # Per-episode action number of the inference active then, so a
                    # tile lines up with exec_action_n in actions.csv and [action
                    # #N] in the log. (rows here always executed >=1 action.)
                    label += f" #{nearest['executed_action_ns'][0]}"
                draw = ImageDraw.Draw(img)
                draw.rectangle([0, 0, img.width, 15], fill=(0, 0, 0))
                draw.text((2, 2), label, fill=(255, 255, 255))
                tiles.append(img)

            cols = max(1, int(np.ceil(np.sqrt(len(tiles)))))
            n_rows = int(np.ceil(len(tiles) / cols))
            tw, th = tiles[0].size
            sheet = Image.new("RGB", (cols * tw, n_rows * th), (30, 30, 30))
            for i, tile in enumerate(tiles):
                sheet.paste(tile, ((i % cols) * tw, (i // cols) * th))
            name = "recap.png"
            sheet.save(self._vla_input_dir / name)
            return name
        except Exception as e:  # noqa: BLE001 - never let recap break an episode
            self.logger.error(f"Failed to render episode recap: {e}")
            return None

    def get_episode_clip(self, num_frames: int) -> tuple[list, float]:
        """Return up to num_frames frames sampled evenly *in time* across the episode
        buffer, oldest-first, along with the wall-clock span those frames cover.

        The buffer decimates rather than evicting its head (see _buffer_clip_frame),
        so it spans the whole episode; the span here is the difference between the
        first and last retained frame timestamps. Callers derive the clip's frame rate
        from this span — using the episode duration instead would hand the VLM
        stretched timestamps.

        Frames are chosen by nearest timestamp to evenly-spaced target times, not by
        index: appends happen at irregular (backpressure-gated) intervals, so index-
        even sampling would not be time-even.

        Returns ([], 0.0) if nothing was buffered (e.g. a very short episode), and a
        span of 0.0 when fewer than two frames were retained.
        """
        with self._clip_lock:
            entries = list(self._episode_clip)
        if not entries or num_frames <= 0:
            return [], 0.0

        span = entries[-1][0] - entries[0][0] if len(entries) > 1 else 0.0
        if len(entries) <= num_frames or span <= 0:
            return [frame for _, frame in entries], span

        # Pick the frame whose timestamp is nearest each evenly-spaced target time,
        # inclusive of both ends. De-duplicate while preserving order so a sparse
        # buffer can't return the same frame twice.
        t0 = entries[0][0]
        timestamps = [ts for ts, _ in entries]
        idxs: list[int] = []
        for k in range(num_frames):
            target = t0 + span * k / (num_frames - 1)
            nearest = min(range(len(entries)), key=lambda i: abs(timestamps[i] - target))
            if nearest not in idxs:
                idxs.append(nearest)
        idxs.sort()
        return [entries[i][1] for i in idxs], span

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def stop(self):
        self.shutdown_event.set()
        self.episode_done.set()
        self.robot.disconnect()
        self.channel.close()
        for handler in list(self.action_logger.handlers):
            self.action_logger.removeHandler(handler)
            handler.close()
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