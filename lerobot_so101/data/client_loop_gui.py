# client_loop_gui.py
#
# Browser front-end for robot_client_loop.py: the same persistent VLA loop (the
# policy server connection and model stay loaded, the arm homes between
# episodes), driven from a web page instead of the terminal prompt. The
# orchestrator's counterpart, without any VLM, is orchestrator_gui.py.
#
#   - instruction box + episode length + Run (one episode at a time; the arm
#     returns home after each, as in the terminal loop)
#   - Stop (ends the running episode early, like Enter in the terminal) and
#     Home (parks the arm while idle)
#   - live camera feeds, live log, the episode list with how each one ended,
#     and recent instructions to re-run; Clear history empties all of them
#
# The task loop stays on the main thread so Ctrl+C in the terminal shuts down
# as before.
#
# Camera feeds: during an episode the page shows the frames the control loop
# already reads; otherwise it polls the robot at --gui_idle_fps.
#
# Logging: nothing is written to disk by default (no action logs, VLA-input
# episodes or logs/loop_client_*.log); pass --no_logging=false to keep them.
# The page's log panel is unaffected.
#
# Usage: robot_client_loop.py's flags, plus optionally --gui_host / --gui_port:
#   python client_loop_gui.py --robot.type=so101_follower ... --gui_port=7861
# then open http://localhost:7861 (the container uses host networking).
# Bind --gui_host=0.0.0.0 only on a trusted network: the page moves the robot.

import json
import logging
import queue
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import draccus
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from lerobot.utils.import_utils import register_third_party_plugins

from gui_common import CameraTap, LogBuffer, camera_stream, idle_camera_poller, mirror_terminal
from robot_client_loop import LoopClientConfig, LoopRobotClient, drop_file_logging

GUI_HTML = Path(__file__).with_name("client_loop_gui.html")
MAX_EPISODE_S = 300.0


@dataclass
class ClientGuiConfig(LoopClientConfig):
    gui_host: str = "127.0.0.1"
    gui_port: int = 7861
    # Robot polling rate for the camera feeds while no episode is running.
    gui_idle_fps: float = 5.0
    # Max rate the camera feeds are copied out of the control loop / streamed.
    gui_stream_fps: float = 15.0
    gui_jpeg_quality: int = 75
    # Write nothing to disk (no action logs, VLA-input episodes or log files);
    # --no_logging=false keeps robot_client_loop.py's logging.
    no_logging: bool = True


class ClientState:
    def __init__(self):
        self.lock = threading.Lock()
        self.busy = False
        self.phase = "starting"    # starting | idle | executing | homing
        self.task = ""
        self.duration = 0.0        # length of the running episode
        self.started = 0.0         # time.monotonic() when it started
        self.stop_requested = False
        self.episodes: list[dict] = []   # finished episodes, oldest first
        self.cleared = 0   # bumped by Clear history; the page empties its log on a change

    def set_phase(self, phase: str):
        with self.lock:
            self.phase = phase

    def snapshot(self) -> dict:
        with self.lock:
            running = self.phase == "executing"
            recent: list[str] = []
            for ep in reversed(self.episodes):
                if ep["task"] not in recent:
                    recent.append(ep["task"])
            return {"busy": self.busy, "phase": self.phase, "task": self.task,
                    "duration": self.duration,
                    "elapsed": time.monotonic() - self.started if running else None,
                    "episodes": list(self.episodes), "recent": recent[:12],
                    "cleared": self.cleared}


def make_client_cls(tap: CameraTap, robot_lock: threading.RLock):
    class GuiClient(LoopRobotClient):
        """LoopRobotClient that hands every observation it reads to the GUI's camera
        tap. Every method that reads or moves the robot holds robot_lock, so the
        idle camera poller never shares the bus with it."""

        def _capture_input(self, obs, task, timestep, elapsed, must_go):
            super()._capture_input(obs, task, timestep, elapsed, must_go)
            tap.offer(obs)

        def capture_frame(self) -> dict:
            with robot_lock:
                obs = super().capture_frame()
            tap.offer(obs, force=True)
            return obs

        def go_home(self):
            with robot_lock:
                super().go_home()

        def run_episode(self, task, enable_abort_listener=True):
            with robot_lock:
                return super().run_episode(task, enable_abort_listener=enable_abort_listener)

    return GuiClient


def how_it_ended(result, duration: float, stopped: bool) -> str:
    if stopped:
        return "stopped"
    if result.converged:
        return "converged"
    if result.duration >= duration - 0.1:
        return "timeout"
    return "ended early"


# ---------------------------------------------------------------------------
# Web app
# ---------------------------------------------------------------------------
class RunRequest(BaseModel):
    task: str
    duration: float | None = None


def build_app(cfg: ClientGuiConfig, client: LoopRobotClient, state: ClientState,
              tap: CameraTap, logs: LogBuffer, jobs: "queue.Queue[tuple]") -> FastAPI:
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return GUI_HTML.read_text(encoding="utf-8")

    @app.get("/api/state")
    def get_state():
        snap = state.snapshot()
        snap["cameras"] = tap.keys()
        snap["config"] = {"policy": cfg.pretrained_name_or_path, "policy_type": cfg.policy_type,
                          "episode_duration": cfg.episode_duration,
                          "max_episode_duration": MAX_EPISODE_S,
                          "logging": "off" if cfg.no_logging else
                                     (cfg.vla_input_dir if cfg.log_vla_inputs else "actions only")}
        return Response(json.dumps(snap, default=str), media_type="application/json")

    @app.get("/api/log")
    def get_log(after: int = 0):
        seq, lines = logs.since(after)
        return {"seq": seq, "lines": lines}

    @app.post("/api/run")
    def run(req: RunRequest):
        task = req.task.strip()
        if not task:
            raise HTTPException(400, "empty instruction")
        duration = cfg.episode_duration if req.duration is None else float(req.duration)
        if not 0 < duration <= MAX_EPISODE_S:
            raise HTTPException(400, f"episode length must be in (0, {MAX_EPISODE_S:g}] s")
        with state.lock:
            if state.busy:
                raise HTTPException(409, "an episode is already running")
            state.busy = True
        jobs.put(("run", task, duration))
        return {"ok": True}

    @app.post("/api/stop")
    def stop():
        with state.lock:
            if state.phase != "executing":
                raise HTTPException(409, "no episode is running")
            state.stop_requested = True
        print("[GUI] STOP requested — ending the episode...")
        client.episode_done.set()
        return {"ok": True}

    @app.post("/api/home")
    def home():
        with state.lock:
            if state.busy:
                raise HTTPException(409, "the robot is busy")
            state.busy = True
        jobs.put(("home",))
        return {"ok": True}

    @app.post("/api/clear")
    def clear():
        with state.lock:
            if state.busy:
                raise HTTPException(409, "wait for the episode to finish")
            logs.clear()
            state.episodes = []
            state.cleared += 1
        return {"ok": True}

    @app.get("/api/camera/{key}")
    def camera(key: str):
        return camera_stream(tap, key, cfg.gui_stream_fps, cfg.gui_jpeg_quality)

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_one(client: LoopRobotClient, state: ClientState, task: str, duration: float):
    """One episode then home, as one turn of robot_client_loop.main; recorded in
    state.episodes."""
    logger = client.logger
    client.config.episode_duration = duration
    with state.lock:
        state.phase, state.task, state.duration = "executing", task, duration
        state.started, state.stop_requested = time.monotonic(), False
    rec = {"n": len(state.episodes) + 1, "task": task, "time": time.strftime("%H:%M:%S"),
           "length": duration}
    try:
        logger.info(f"Starting episode: '{task}'")
        result = client.run_episode(task, enable_abort_listener=False)
        rec.update(asdict(result))
        rec["ended"] = how_it_ended(result, duration, state.stop_requested)
    except Exception as e:  # noqa: BLE001 - keep serving after an episode crash
        logger.exception(f"Episode '{task}' crashed: {e}")
        rec.update(ended="crashed", error=repr(e))
    state.set_phase("homing")
    try:
        client.go_home()
    except Exception:  # noqa: BLE001
        logger.exception("go_home after the episode failed")
    with state.lock:
        state.episodes.append(rec)


@draccus.wrap()
def main(cfg: ClientGuiConfig):
    logging.basicConfig(level=logging.INFO)

    logs = LogBuffer()
    restore_terminal = mirror_terminal(logs)
    if cfg.no_logging:
        cfg.log_actions = cfg.log_vla_inputs = cfg.save_vla_inputs = False
        drop_file_logging()
    tap = CameraTap(cfg.gui_stream_fps)
    state = ClientState()
    robot_lock = threading.RLock()

    client = make_client_cls(tap, robot_lock)(cfg)
    logger = client.logger
    if not client.connect_server():
        logger.error("Could not connect to policy server. Exiting.")
        client.stop()
        restore_terminal()
        return

    jobs: "queue.Queue[tuple]" = queue.Queue()
    stop = threading.Event()
    server = uvicorn.Server(uvicorn.Config(build_app(cfg, client, state, tap, logs, jobs),
                                           host=cfg.gui_host, port=cfg.gui_port,
                                           log_level="warning"))
    threading.Thread(target=server.run, daemon=True, name="gui-server").start()
    threading.Thread(target=idle_camera_poller, daemon=True, name="gui-camera",
                     args=(client.robot, robot_lock, tap, cfg.gui_idle_fps, stop)).start()

    logger.info("=" * 60)
    logger.info("Policy server connected. Model loaded and ready.")
    logger.info(f"GUI: http://{cfg.gui_host}:{cfg.gui_port}  (Ctrl+C here to shut down)"
                + ("  — no logging" if cfg.no_logging else ""))
    logger.info("=" * 60)
    state.set_phase("idle")

    try:
        while True:
            try:
                job = jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            if job[0] == "run":
                run_one(client, state, job[1], job[2])
            else:
                state.set_phase("homing")
                try:
                    client.go_home()
                except Exception:  # noqa: BLE001
                    logger.exception("go_home failed")
            with state.lock:
                state.busy, state.phase = False, "idle"
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
    finally:
        stop.set()
        server.should_exit = True
        client.episode_done.set()
        with robot_lock:
            client.go_home()
            client.stop()
        restore_terminal()


if __name__ == "__main__":
    register_third_party_plugins()
    main()
