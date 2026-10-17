# orchestrator_gui.py
#
# Browser front-end for vlm_robot_orchestrator.py: the same VLM planner + VLA
# loop, driven from a web page instead of the terminal prompt.
#
#   - prompt box + Run (one high-level task at a time)
#   - Skip (ends the running sub-task's episode) and Abort (abandons the whole
#     task and parks the arm) for operator control during a demo
#   - live plan: each round's sub-tasks with their status, every episode's VLM
#     judgement and monitor outcome, the final verdict, past tasks
#   - live log (everything the terminal shows) and live camera feeds
#   - VLM output: the parsed result of every VLM step (plan, monitor checks,
#     sub-task eval, final eval, replan), recorded by wrapping the planner's and
#     the monitor's entry points
#
# The server runs inside the orchestrator process (it owns the robot, the policy
# connection and the loaded VLM); the task loop stays on the main thread so
# Ctrl+C in the terminal shuts down as before. 'q'+Enter in the terminal still
# aborts a running task. Replanning is not exposed.
#
# Camera feeds: during a task the page shows the frames the control loop / VLM
# already read (the GUI never reads the robot itself mid-task); between tasks it
# polls the robot at --gui_idle_fps.
#
# Usage: the orchestrator's flags, plus optionally --gui_host / --gui_port:
#   python orchestrator_gui.py --robot.type=so101_follower ... --gui_port=7860
# then open http://localhost:7860 (the container uses host networking).
# Bind --gui_host=0.0.0.0 only on a trusted network: the page moves the robot.

import copy
import functools
import inspect
import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import draccus
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel

from lerobot.utils.import_utils import register_third_party_plugins

from vlm_monitor_live import TappedClient
from vlm_robot_orchestrator import OrchestratorConfig, open_session

GUI_HTML = Path(__file__).with_name("orchestrator_gui.html")


@dataclass
class GuiConfig(OrchestratorConfig):
    gui_host: str = "127.0.0.1"
    gui_port: int = 7860
    # Robot polling rate for the camera feeds while no task is running.
    gui_idle_fps: float = 5.0
    # Max rate the camera feeds are copied out of the control loop / streamed.
    gui_stream_fps: float = 15.0
    gui_jpeg_quality: int = 75


# ---------------------------------------------------------------------------
# Shared state between the task loop, the robot client and the web server
# ---------------------------------------------------------------------------
class LogBuffer:
    """Terminal mirror (a _Tee sink) keeping the last `maxlen` lines, numbered so
    the page can poll for only what is new. A '\\r' (progress bars) rewrites the
    current partial line rather than adding lines."""

    def __init__(self, maxlen: int = 5000):
        self._lines: deque[tuple[int, str]] = deque(maxlen=maxlen)
        self._seq = 0
        self._partial = ""
        self._lock = threading.Lock()

    def write(self, data: str):
        with self._lock:
            text = self._partial + data
            *done, self._partial = text.split("\n")
            for line in done:
                line = line.rsplit("\r", 1)[-1]
                self._seq += 1
                self._lines.append((self._seq, line))

    def flush(self):
        pass

    def since(self, after: int) -> tuple[int, list[str]]:
        with self._lock:
            lines = [l for s, l in self._lines if s > after]
            return self._seq, lines


class CameraTap:
    """Latest frame per camera, as copies (the camera loop reuses its arrays),
    taken at most every 1/fps seconds so the control loop pays little for it."""

    def __init__(self, fps: float):
        self._period = 1.0 / max(fps, 0.1)
        self._frames: dict[str, np.ndarray] = {}
        self._seq: dict[str, int] = {}
        self._last = 0.0
        self._lock = threading.Lock()

    def offer(self, obs: dict, force: bool = False):
        now = time.monotonic()
        if not force and now - self._last < self._period:
            return
        self._last = now
        frames = TappedClient._select_all_camera_frames(obs)
        with self._lock:
            for key, arr in frames.items():
                self._frames[key] = np.array(arr, dtype=np.uint8, copy=True)
                self._seq[key] = self._seq.get(key, 0) + 1

    def keys(self) -> list[str]:
        with self._lock:
            return sorted(self._frames)

    def get(self, key: str) -> tuple[int, np.ndarray | None]:
        with self._lock:
            return self._seq.get(key, 0), self._frames.get(key)


def _yes(value) -> bool | None:
    return None if value is None else bool(value)


def _sum_list(noun: str):
    def summarise(result, _args) -> dict:
        return {"verdict": None, "text": f"{len(result or [])} {noun}(s)",
                "items": [str(x) for x in result or []]}
    return summarise


def _sum_judgement(result, _args) -> dict:
    tags = [result["failure_type"]] if result.get("failure_type") and not result.get("success") else []
    return {"verdict": _yes(result.get("success")), "text": str(result.get("reason", "")),
            "items": [], "tags": tags}


def _sum_precheck(result, _args) -> dict:
    return {"verdict": _yes(result.get("done")), "text": str(result.get("reason", "")),
            "items": []}


def _sum_final(result, _args) -> dict:
    items = [f"{f['object']}: {f.get('before', '?')} → {f.get('support', '?')}"
             for f in result.get("findings") or []]
    if not items:
        items = [f"{r.get('object')} → {r.get('support')}" for r in result.get("surveyed") or []]
    s = result.get("sets") or {}
    if s:
        items.append(f"must be on tray: {s.get('must_be_on_tray', [])}")
        if s.get("must_not_be_on_tray"):
            items.append(f"must not be on tray: {s['must_not_be_on_tray']}")
        if s.get("choose_from"):
            items.append(f"{s.get('choose_count')} of {s['choose_from']}")
        if s.get("unmatched"):
            items.append(f"unmatched names: {s['unmatched']}")
    return {"verdict": _yes(result.get("success")), "text": str(result.get("reason", "")),
            "items": items}


def _sum_reach(result, _args) -> dict | None:
    if not result:
        return {"verdict": None, "text": "unparseable", "items": []}
    if str(result.get("reaching_for", "")).startswith("none (arm not moving)"):
        return None  # motion gate: no VLM call was made
    action = str(result.get("action", "?")).upper()
    return {"verdict": action != "STOP", "text": str(result.get("reason", "")),
            "items": [f"reaching for: {result.get('reaching_for')}",
                      f"match: {result.get('match')}"], "tags": [action]}


def _sum_held(result, _args) -> dict:
    held, answers = result
    return {"verdict": None, "text": f"holding: {held}",
            "items": [f"frame {a['frame']}: {a['object']}" for a in answers]}


def _sum_grounding(result, _args) -> dict:
    return {"verdict": None, "text": f"{result.get('object')} — {result.get('reason', '')}",
            "items": []}


def _sum_inventory(result, _args) -> dict:
    return {"verdict": None, "text": f"{len(result or [])} object(s)",
            "items": [f"{o['name']} ({o.get('appearance', '')})" for o in result or []]}


class VlmResultLog:
    """The parsed result of every VLM step, for the page's VLM panel.

    Installed by wrapping the planner's entry points (VLMPlanner methods and the
    two-pass identify step) and the monitor's VLM checks (vlm_monitor_tune
    functions, looked up through the module by vlm_monitor_live). Each wrapped
    call adds an entry when it starts (shown as running) and fills in a summary
    — verdict, reason, short items — when it returns; each change bumps `ver` so
    the page polls for only what changed. A summariser returning None drops the
    entry (a monitor check the motion gate answered without the VLM)."""

    def __init__(self, maxlen: int = 400):
        self._entries: deque[dict] = deque(maxlen=maxlen)
        self._ver = 0
        self._next_id = 0
        self._lock = threading.Lock()

    def install(self):
        import vlm_monitor_tune as T
        import vlm_robot_orchestrator as orch

        P = orch.VLMPlanner
        for owner, name, category, step, input_arg, summarise in [
            (orch, "identify_objects", "plan", "objects", None, _sum_list("object")),
            (P, "decompose", "plan", "plan", "prompt", _sum_list("sub-task")),
            (P, "replan", "replan", "replan", "failed_subtask", _sum_list("sub-task")),
            (P, "refine_label", "replan", "refine label", "failed_instruction", _sum_list("name")),
            (P, "precheck_subtask_done", "subtask", "pre-check", "sub_task", _sum_precheck),
            (P, "evaluate", "subtask", "sub-task eval", "sub_task", _sum_judgement),
            (P, "evaluate_final", "final", "final eval", "original_prompt", _sum_final),
            (T, "run_check", "monitor", "reach check", "sub_task", _sum_reach),
            (T, "verify_held", "monitor", "wrist check", None, _sum_held),
            (T, "match_task_object", "monitor", "grounding", "task", _sum_grounding),
            (T, "scene_inventory", "monitor", "inventory", None, _sum_inventory),
        ]:
            setattr(owner, name, self._wrap(getattr(owner, name), category, step,
                                            input_arg, summarise))

    def _wrap(self, fn, category, step, input_arg, summarise):
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def recorded(*args, **kwargs):
            subject = ""
            if input_arg:
                try:
                    subject = str(sig.bind_partial(*args, **kwargs).arguments.get(input_arg) or "")
                except TypeError:
                    pass
            entry = self._start(category, step, subject)
            t0 = time.monotonic()
            try:
                result = fn(*args, **kwargs)
            except BaseException as e:
                self._finish(entry, {"verdict": None, "text": "", "items": []},
                             time.monotonic() - t0, error=repr(e))
                raise
            try:
                summary = summarise(result, args)
            except Exception as e:  # noqa: BLE001 - an odd result must not break the run
                summary = {"verdict": None, "text": f"(could not summarise: {e})", "items": []}
            self._finish(entry, summary, time.monotonic() - t0)
            return result

        return recorded

    def _start(self, category: str, step: str, subject: str) -> dict:
        with self._lock:
            self._next_id += 1
            self._ver += 1
            entry = {"id": self._next_id, "ver": self._ver, "time": time.strftime("%H:%M:%S"),
                     "category": category, "step": step, "subject": subject, "done": False,
                     "dropped": False, "verdict": None, "text": "", "items": [], "tags": [],
                     "error": None, "seconds": None}
            self._entries.append(entry)
            return entry

    def _finish(self, entry: dict, summary: dict | None, seconds: float,
                error: str | None = None):
        with self._lock:
            self._ver += 1
            if summary is None:
                entry.update(dropped=True, ver=self._ver)
                return
            entry.update(done=True, error=error, seconds=round(seconds, 2), ver=self._ver,
                         verdict=summary.get("verdict"), text=summary.get("text", ""),
                         items=summary.get("items", []), tags=summary.get("tags", []))

    def since(self, after: int) -> tuple[int, list[dict]]:
        with self._lock:
            return self._ver, [dict(e) for e in self._entries if e["ver"] > after]


class GuiState:
    def __init__(self):
        self.lock = threading.Lock()
        self.busy = False
        self.phase = "starting"    # starting | idle | planning | executing | evaluating
        self.phase_detail = ""
        self.trace: dict | None = None
        self.history: list[dict] = []   # finished tasks, oldest first

    def set_phase(self, phase: str, detail: str = ""):
        with self.lock:
            self.phase, self.phase_detail = phase, detail

    def snapshot(self) -> dict:
        with self.lock:
            trace = self.trace
            out = {"busy": self.busy, "phase": self.phase, "phase_detail": self.phase_detail,
                   "history": list(self.history)}
        # The trace is filled in place by the task thread; its dicts keep fixed keys,
        # but retry the copy in case one is caught mid-change.
        for _ in range(3):
            try:
                out["trace"] = copy.deepcopy(trace)
                break
            except RuntimeError:
                time.sleep(0.01)
        else:
            out["trace"] = None
        return out


def make_client_cls(tap: CameraTap, state: GuiState):
    class GuiClient(TappedClient):
        """TappedClient that hands every observation it reads to the GUI's camera
        tap and reports when an episode is running."""

        def _capture_input(self, obs, task, timestep, elapsed, must_go):
            super()._capture_input(obs, task, timestep, elapsed, must_go)
            tap.offer(obs)

        def capture_frame(self) -> dict:
            obs = super().capture_frame()
            tap.offer(obs, force=True)
            return obs

        def run_episode(self, task, enable_abort_listener=True):
            state.set_phase("executing", task)
            try:
                return super().run_episode(task, enable_abort_listener=enable_abort_listener)
            finally:
                state.set_phase("evaluating", task)

    return GuiClient


# ---------------------------------------------------------------------------
# Web app
# ---------------------------------------------------------------------------
class RunRequest(BaseModel):
    prompt: str


def build_app(cfg: GuiConfig, session, state: GuiState, tap: CameraTap, logs: LogBuffer,
              vlm_calls: VlmResultLog, jobs: "queue.Queue[str]") -> FastAPI:
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return GUI_HTML.read_text(encoding="utf-8")

    @app.get("/api/state")
    def get_state():
        snap = state.snapshot()
        snap["cameras"] = tap.keys()
        snap["run_dir"] = str(session.run_dir)
        snap["config"] = {"policy": cfg.pretrained_name_or_path, "vlm": cfg.vlm_model,
                          "episode_duration": cfg.episode_duration,
                          "max_retries": cfg.max_retries,
                          "max_final_retries": cfg.max_final_retries,
                          "monitor": cfg.enable_vlm_monitor}
        return Response(json.dumps(snap, default=str), media_type="application/json")

    @app.get("/api/log")
    def get_log(after: int = 0):
        seq, lines = logs.since(after)
        return {"seq": seq, "lines": lines}

    @app.get("/api/vlm")
    def get_vlm(after: int = 0):
        ver, entries = vlm_calls.since(after)
        return Response(json.dumps({"ver": ver, "entries": entries}, default=str),
                        media_type="application/json")

    @app.post("/api/run")
    def run(req: RunRequest):
        prompt = req.prompt.strip()
        if not prompt:
            raise HTTPException(400, "empty prompt")
        with state.lock:
            if state.busy:
                raise HTTPException(409, "a task is already running")
            state.busy = True
        jobs.put(prompt)
        return {"ok": True}

    @app.post("/api/skip")
    def skip():
        with state.lock:
            executing = state.busy and state.phase == "executing"
        if not executing:
            raise HTTPException(409, "no sub-task is executing")
        print("[GUI] SKIP requested — stopping current action...")
        session.interjection.request_skip()
        return {"ok": True}

    @app.post("/api/abort")
    def abort():
        with state.lock:
            busy = state.busy
        if not busy:
            raise HTTPException(409, "no task is running")
        print("[GUI] ABORT requested — stopping current action...")
        session.interjection.request_abort()
        return {"ok": True}

    @app.get("/api/camera/{key}")
    def camera(key: str):
        period = 1.0 / max(cfg.gui_stream_fps, 0.1)
        params = [cv2.IMWRITE_JPEG_QUALITY, int(cfg.gui_jpeg_quality)]

        def stream():
            # Re-send the last frame every few seconds even when nothing is new
            # (e.g. a long VLM call): a closed tab is only noticed at a yield, so
            # this bounds how long it holds a server thread.
            last, last_sent = -1, 0.0
            while True:
                seq, frame = tap.get(key)
                if frame is not None and (seq != last or time.monotonic() - last_sent > 2.0):
                    last, last_sent = seq, time.monotonic()
                    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                           params)
                    if ok:
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                               + jpg.tobytes() + b"\r\n")
                time.sleep(period)

        return StreamingResponse(stream(),
                                 media_type="multipart/x-mixed-replace; boundary=frame")

    return app


def idle_camera_poller(session, robot_lock: threading.Lock, tap: CameraTap,
                       fps: float, stop: threading.Event):
    """Keep the feeds live between tasks. Reads the robot only while it holds
    robot_lock, which the task loop holds for the whole of every task."""
    period = 1.0 / max(fps, 0.1)
    while not stop.wait(period):
        if not robot_lock.acquire(blocking=False):
            continue
        try:
            tap.offer(session.client.robot.get_observation(), force=True)
        except Exception as e:  # noqa: BLE001 - a dropped frame must not kill the feed
            logging.debug(f"GUI idle camera read failed: {e}")
        finally:
            robot_lock.release()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@draccus.wrap()
def main(cfg: GuiConfig):
    logging.basicConfig(level=logging.INFO)

    logs = LogBuffer()
    tap = CameraTap(cfg.gui_stream_fps)
    state = GuiState()
    session = open_session(cfg, client_cls=make_client_cls(tap, state), mirrors=(logs,))
    if session is None:
        return
    logger = session.client.logger
    vlm_calls = VlmResultLog()
    vlm_calls.install()

    jobs: "queue.Queue[str]" = queue.Queue()
    robot_lock = threading.Lock()
    stop = threading.Event()

    server = uvicorn.Server(uvicorn.Config(build_app(cfg, session, state, tap, logs, vlm_calls, jobs),
                                           host=cfg.gui_host, port=cfg.gui_port,
                                           log_level="warning"))
    threading.Thread(target=server.run, daemon=True, name="gui-server").start()
    threading.Thread(target=idle_camera_poller, daemon=True, name="gui-camera",
                     args=(session, robot_lock, tap, cfg.gui_idle_fps, stop)).start()

    logger.info("=" * 60)
    logger.info("Dual-system ready: VLM planner + VLA policy loaded.")
    logger.info(f"GUI: http://{cfg.gui_host}:{cfg.gui_port}  (Ctrl+C here to shut down; "
                f"'q'+Enter here aborts a running task)")
    logger.info("=" * 60)
    state.set_phase("idle")

    def on_trace(trace: dict):
        with state.lock:
            state.trace = trace

    try:
        while True:
            try:
                prompt = jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            with state.lock:
                state.trace = None
            state.set_phase("planning", prompt)
            trace = None
            with robot_lock:
                try:
                    # Terminal listener in abort-only mode: skip comes from the page,
                    # replanning is not offered.
                    trace = session.run(prompt, abort_only=True, on_trace=on_trace)
                except Exception as e:  # noqa: BLE001 - keep serving after a task crash
                    logger.exception(f"Task '{prompt}' crashed: {e}")
                    try:
                        session.client.go_home()
                    except Exception:  # noqa: BLE001
                        logger.exception("go_home after the crash failed")
            with state.lock:
                done = trace or state.trace or {"prompt": prompt}
                state.history.append({
                    "prompt": prompt, "outcome": done.get("outcome") or "crashed",
                    "final_success": done.get("final_success"),
                    "final_reason": done.get("final_reason", ""),
                    "task_dir": done.get("task_dir"), "finished": time.strftime("%H:%M:%S")})
                state.busy = False
                state.phase, state.phase_detail = "idle", ""
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
    finally:
        stop.set()
        server.should_exit = True
        with robot_lock:
            session.close()


if __name__ == "__main__":
    register_third_party_plugins()
    main()
