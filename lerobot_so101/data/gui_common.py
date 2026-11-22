# gui_common.py
#
# Pieces shared by the browser front-ends (orchestrator_gui.py for the VLM
# orchestrator, client_loop_gui.py for the plain robot client loop): the
# terminal mirror behind the page's log panel, the camera tap that copies frames
# out of the control loop, the MJPEG stream the page's <img> tags read, and the
# idle poller that keeps the feeds live while no episode runs.

import logging
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
from fastapi.responses import StreamingResponse

from robot_client_loop import LoopRobotClient


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

    def clear(self):
        """Drop the kept lines; numbering carries on, so pollers stay in step."""
        with self._lock:
            self._lines.clear()


class _Tee:
    """Write a text stream through to the terminal and to every sink."""

    def __init__(self, primary, *sinks):
        self._primary = primary
        self._sinks = sinks

    def write(self, data):
        self._primary.write(data)
        for sink in self._sinks:
            sink.write(data)
        return len(data)

    def flush(self):
        self._primary.flush()
        for sink in self._sinks:
            sink.flush()

    def isatty(self):
        return self._primary.isatty()

    def fileno(self):
        return self._primary.fileno()

    def __getattr__(self, name):
        return getattr(self._primary, name)


def mirror_terminal(*sinks):
    """Copy everything written to stdout/stderr (prints and logging) into `sinks`.
    Returns a function that restores the original streams."""
    orig = (sys.stdout, sys.stderr)
    sys.stdout = _Tee(orig[0], *sinks)
    sys.stderr = _Tee(orig[1], *sinks)
    # The root logger's console handler grabbed the original stderr when it was
    # created, so re-point it at the tee; file handlers stay as they are.
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            h.setStream(sys.stderr)

    def restore():
        sys.stdout, sys.stderr = orig

    return restore


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
        frames = LoopRobotClient._select_all_camera_frames(obs)
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


def camera_stream(tap: CameraTap, key: str, fps: float, jpeg_quality: int) -> StreamingResponse:
    """MJPEG stream of one camera from the tap, at most `fps` frames a second."""
    period = 1.0 / max(fps, 0.1)
    params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]

    def stream():
        # Re-send the last frame every few seconds even when nothing is new
        # (e.g. a long VLM call): a closed tab is only noticed at a yield, so
        # this bounds how long it holds a server thread.
        last, last_sent = -1, 0.0
        while True:
            seq, frame = tap.get(key)
            if frame is not None and (seq != last or time.monotonic() - last_sent > 2.0):
                last, last_sent = seq, time.monotonic()
                ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), params)
                if ok:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                           + jpg.tobytes() + b"\r\n")
            time.sleep(period)

    return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")


def idle_camera_poller(robot, robot_lock: threading.RLock, tap: CameraTap,
                       fps: float, stop: threading.Event):
    """Keep the feeds live while no episode runs (between tasks and during VLM
    calls). Reads the robot only while it holds robot_lock, which the client
    holds for every episode, homing move and frame capture."""
    period = 1.0 / max(fps, 0.1)
    while not stop.wait(period):
        if not robot_lock.acquire(blocking=False):
            continue
        try:
            tap.offer(robot.get_observation(), force=True)
        except Exception as e:  # noqa: BLE001 - a dropped frame must not kill the feed
            logging.debug(f"GUI idle camera read failed: {e}")
        finally:
            robot_lock.release()
