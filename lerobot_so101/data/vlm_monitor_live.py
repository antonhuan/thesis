"""
Live (physical robot) REPL for the semantic monitor — the robot counterpart of
vlm_monitor_test.py.

You control the two task strings separately, exactly like the offline correct /
wrong-object trials:
  /vla <task>   what the VLA policy is told to do            (drives the arm)
  /vlm <task>   what the monitor believes the sub-task is    (what it checks against)
A CORRECT trial gives both the same object; a WRONG trial tells the monitor a different
object than the one the VLA is going for. After each episode you label what actually
happened and a running specificity / recall tally is kept.

The monitor runs the SAME decision flow that was tested offline (vlm_monitor_tune), made
causal for a live stream:
  * frames: every VLA input the client captures while the arm is moving (front + wrist,
    joints) — the same cadence as the offline episodes — slid in a --window-frame window;
  * gate (default 'grasp' — ONE CHECK PER RUN): the monitor stays silent until the grasp. A
    per-control-tick watch freezes the arm the instant the gripper has closed on an object
    (before the lift); ONE look-back reach check then judges the last --window frames of
    the approach; a STOP is confirmed / vetoed by the wrist probe on two fresh still frames
    taken during the freeze, and an 'unclear' reach is decided by the wrist
    (--unclear_wrist_decides); then the monitor retires. Fallback: one check if the gripper
    opened but no grasp followed within --grasp_fallback_frames. Offline (73 episodes):
    91.0% balanced accuracy with ~1 reach check per trial (hybrid_grasp_uw).
    Other gates: 'distance' (joint start/still gate, several checks, 95.1% offline but it
    stops only after the lift) and 'close' (open when the live queue plans the close
    within --horizon steps, until --tail frames after the first close);
  * two-pass reach check (run_check) with the kinematic 'held', and the grasp commit;
  * hybrid verify (--verify, default on): a STOP is a suspicion confirmed at the grasp — if
    raised before the gripper closes, the arm is allowed to reach the close, then the
    wrist probe on the close frame and the next one vetoes (object == grounded target) or
    confirms it. Grounding: 'scene' (closed object list, --objects) or 'inventory' (a
    vocabulary built once from the initial scene).
  * the arm is PAUSED (execution gate) while the VLM runs.

Each trial writes <results_dir>/trial_NNN.json (tasks, config, per-window decisions,
verify details, label) and the client saves the episode in the offline format
(frames + actions.csv) under --vla_input_dir, so a live trial can be replayed offline
with vlm_monitor_tune.py / vlm_monitor_test.py.

Usage (same robot / policy flags as robot_client_loop.py):
    python vlm_monitor_live.py \\
        --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=arm \\
        --robot.cameras="{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, front: {type: intelrealsense, serial_number_or_name: 342222071104, width: 640, height: 480, fps: 30}}" \\
        --task=dummy --server_address=127.0.0.1:8080 --policy_type=pi05 \\
        --pretrained_name_or_path=ant0nh/pi05_pnp_425_25k --policy_device=cuda \\
        --actions_per_chunk=50 --chunk_size_threshold=0.7 --episode_duration=100 \\
        --gate=grasp --match=inventory
With --match=inventory (default) nothing about the scene is hard-coded: the object list is
built from the initial frame each episode. --match=scene uses the fixed --objects list.
"""

import contextlib
import io
import json
import logging
import queue
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import draccus
import numpy as np
from PIL import Image

from robot_client_loop import LoopClientConfig, LoopRobotClient
import vlm_monitor_tune as T

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LiveMonitorConfig(LoopClientConfig):
    vlm_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    temperature: float = 0.1
    # Initial task strings (change them in the REPL with /vla, /vlm, /both).
    vla_task: str = "put the pouch on the tray"
    vlm_task: str = "put the pouch on the tray"
    # Monitor settings — defaults are the best offline configuration.
    window: int = 4
    gate: str = "grasp"            # grasp (one check per run) | close | distance
    # gate=grasp: stay gated until the grasp, freeze ONCE (per-tick grasp watch), judge the
    # look-back window + wrist, retire. Fallback: if the gripper opened but no grasp came
    # within this many frames, check once anyway.
    grasp_fallback_frames: int = T.DEFAULT_GRASP_FALLBACK_FRAMES
    # gate=grasp: when the single look-back check is 'unclear', the wrist (fresh frames at
    # the grasp freeze) decides — a clearly different scene object -> STOP.
    unclear_wrist_decides: bool = True
    horizon: int = T.DEFAULT_APPROACH_GATE_HORIZON
    tail: int = 1
    verify: bool = True
    # Object grounding. 'inventory' (default, OPEN — no hard-coded list): the VLM lists the
    # pickable objects on the initial scene once per episode (name + appearance); the task
    # is grounded to one of them and the wrist probe chooses among the same entries (+
    # 'other' for an object the inventory missed). 'scene': the fixed --objects list.
    match: str = "inventory"          # inventory | scene
    # Ground-compare: give the reach check's pass 2 the sub-task's object grounded on the
    # initial scene (per `match`) instead of the task wording (see run_scan's flag).
    ground_compare: bool = False
    objects: list[str] = field(default_factory=lambda: list(T.HELD_PROBE_OBJECTS))
    wrist_key: str = "wrist"
    wrist_rotate: float = T.DEFAULT_WRIST_ROTATE_CCW
    # Frames to wait for the grasp close after a pre-grasp STOP before the STOP stands
    # unverified (fail-safe).
    pending_timeout_frames: int = 10
    # Observations the client may send to the policy server while the monitor holds
    # the arm paused. The base client re-sends (forced inference) on EVERY control tick
    # while paused, which keeps the shared GPU busy with policy chunks (~0.9 s each) and
    # starves the VLM check — a 2 s check stretched to ~20 s on the robot. One refresh
    # keeps the queue current for the resume.
    pause_refreshes: int = 1
    # End-of-episode release: after ANY termination (monitor STOP, timeout, convergence,
    # abort) the gripper is opened to this position (gripper.pos is 0-100 % of its
    # calibrated range; the policy opens to ~28-38 when grasping) over release_steps
    # interpolation steps, held for release_dwell_s so the object drops free, and then
    # the arm returns to the configured home pose.
    # Grasp freeze: while a pre-grasp STOP is pending, the client watches the gripper on
    # EVERY control tick and freezes the arm the moment it has closed on the object
    # (aperture >= DEFAULT_GRASP_CLOSE_MARGIN below its open peak AND the closing has
    # stalled: < freeze_stall_eps change over the last 3 ticks), before the lift. The
    # wrist verify then uses two FRESH still frames (after freeze_settle_s, >=
    # verify_frame_gap_s apart) instead of the next chunk frame, which arrives ~1 s later
    # with the object already lifting.
    freeze_stall_eps: float = 0.5
    freeze_settle_s: float = 0.3
    verify_frame_gap_s: float = 0.2
    gripper_open_pos: float = 40.0
    release_steps: int = 15
    release_dwell_s: float = 0.5
    results_dir: str = "live_monitor_trials"
    # Episodes are saved in the offline format here (frames + actions.csv).
    vla_input_dir: str = "live_monitor_trials/episodes"


# ---------------------------------------------------------------------------
# Client tap: forward each captured VLA input (while the arm moves) to the monitor
# ---------------------------------------------------------------------------


class TappedClient(LoopRobotClient):
    """LoopRobotClient that forwards to a monitor queue exactly the frames the offline
    episodes contain: the VLA inputs the policy actually PREDICTED from, i.e. the input
    a received action chunk binds to (_bind_chunk_input) — one per chunk, ~0.8 s apart.
    NOT every observation sent: the client sends on most control ticks (~30 Hz) and the
    server predicts only some, so tapping the sends gives a 30 Hz stream whose 4-frame
    window spans ~0.1 s (the arm looks still, the motion gate skips everything). Inputs
    captured while the arm is paused (the forced re-sends during a VLM check) are not
    forwarded either — the offline recordings have no pauses. Requires log_vla_inputs."""

    supports_grasp_watch = True

    def __init__(self, config):
        super().__init__(config)
        self.monitor_inbox: "queue.Queue | None" = None
        self._pause_sends = 0
        # Armed by the monitor while a pre-grasp STOP is pending (see _watch_tick).
        self.grasp_watch: "dict | None" = None
        self._watch_lock = threading.Lock()

    def send_observation(self, obs):
        # While the monitor holds the arm, send at most `pause_refreshes` observations
        # (the base loop would force one every tick and saturate the shared GPU).
        if not self.execution_gate.is_set():
            if self._pause_sends >= getattr(self.config, "pause_refreshes", 1):
                return True
            self._pause_sends += 1
        else:
            self._pause_sends = 0
        return super().send_observation(obs)

    def _capture_input(self, obs, task, timestep, elapsed, must_go):
        super()._capture_input(obs, task, timestep, elapsed, must_go)
        # Tag the held record with whether the arm was running when it was captured (a
        # record that replaced an earlier one for this timestep has no tag yet).
        gate_open = self.execution_gate.is_set()
        with self._vla_lock:
            rec = self._sent_inputs.get(int(timestep))
            if rec is not None and "gate_open" not in rec:
                rec["gate_open"] = gate_open
        if self.grasp_watch is not None:
            self._watch_tick(obs, elapsed)

    def _watch_tick(self, obs, elapsed):
        """Per-control-tick grasp watch (runs in the control loop; the capture hook is
        called on every tick an observation is read, including while paused). Freezes
        the arm as soon as the gripper has closed on the object and stalled, then
        collects two still wrist frames for the verify."""
        cfg = self.config
        g_key = next((k for k in self.robot.action_features if k.split(".")[0] == "gripper"), None)
        if g_key is None:
            return
        obs_key = f"{g_key}.pos" if f"{g_key}.pos" in obs else g_key
        if obs_key not in obs:
            return
        g = float(obs[obs_key])
        with self._watch_lock:
            w = self.grasp_watch
            if w is None or w.get("ready"):
                return
            if w.get("frozen_at") is None:
                w["peak"] = max(w["peak"], g)
                w["hist"] = (w["hist"] + [g])[-4:]
                opened = w["peak"] - w["start"] >= T.GRIPPER_OPEN_MIN
                closed = g <= w["peak"] - T.DEFAULT_GRASP_CLOSE_MARGIN
                stalled = (len(w["hist"]) >= 4
                           and max(w["hist"]) - min(w["hist"]) < cfg.freeze_stall_eps)
                if opened and closed and stalled:
                    self.execution_gate.clear()
                    w["frozen_at"] = float(elapsed)
                    w["aperture"] = g
                    self.logger.info(f"[grasp watch] gripper closed ({g:.1f}, peak "
                                     f"{w['peak']:.1f}) — arm frozen at t={elapsed:.2f}s")
                return
            # Frozen: take two settled wrist (+ front) frames for the verify.
            if elapsed < w["frozen_at"] + cfg.freeze_settle_s:
                return
            if w["frames"] and elapsed < w["frames"][-1]["elapsed"] + cfg.verify_frame_gap_s:
                return
            frames = self._select_all_camera_frames(obs)
            w["frames"].append({"elapsed": float(elapsed), "frames": {
                k: np.asarray(v, dtype=np.uint8).copy() for k, v in frames.items()}})
            if len(w["frames"]) >= 2:
                w["ready"] = True

    def _bind_chunk_input(self, chunk_id, ts0, returned_len):
        super()._bind_chunk_input(chunk_id, ts0, returned_len)
        inbox = self.monitor_inbox
        if inbox is None:
            return
        with self._vla_lock:
            rec = self._active_chunks.get(chunk_id)
            if rec is None or not rec.get("gate_open", True):
                return
            inbox.put({"frames": dict(rec["frames"]), "state": list(rec["state"]),
                       "timestep": int(rec["timestep"]), "elapsed": float(rec["elapsed"])})

    def release_gripper(self):
        """Smoothly open the gripper in place (other joints held), then dwell."""
        cfg = self.config
        current = self._capture_current_position()
        g = next((k for k in self.robot.action_features if k.split(".")[0] == "gripper"), None)
        if g is None:
            self.logger.warning("No gripper joint found; skipping release.")
            return
        start, end = current.get(g, 0.0), float(cfg.gripper_open_pos)
        self.logger.info(f"Releasing: opening gripper {start:.1f} -> {end:.1f}")
        for step in range(1, cfg.release_steps + 1):
            target = dict(current)
            target[g] = start + step / cfg.release_steps * (end - start)
            self.robot.send_action(target)
            time.sleep(cfg.home_step_dt)
        time.sleep(cfg.release_dwell_s)

    def release_and_home(self):
        """End-of-episode sequence: open the gripper, then return to home."""
        self.execution_gate.set()
        self.release_gripper()
        self.go_home()

    def queued_gripper(self, horizon: int) -> list[float]:
        """Gripper targets of the next `horizon` queued actions (the live look-ahead)."""
        names = list(self.robot.action_features)
        g = next((n for n in names if n.split(".")[0] == "gripper"), None)
        if g is None:
            return []
        with self.action_queue_lock:
            items = sorted(self.action_queue.queue, key=lambda ta: ta.get_timestep())
        out = []
        for ta in items[:horizon]:
            d = self._action_tensor_to_action_dict(ta.get_action())
            if g in d:
                out.append(float(d[g]))
        return out


# Inventory entries describing the robot or scene fixtures rather than a pickable object.
# The inventory prompt already excludes them, but the VLM still lists e.g. "orange device
# (orange plastic with wires and label 'KUMQUAT')" — the arm — and grounding can then map
# the task onto it (trial 17). Matched as whole words in the name or appearance; colours
# are deliberately not used (a real orange object must survive).
_FIXTURE_RE = re.compile(r"\b(robot\w*|arm|gripper|servo|motor|device|machine|wires?|cables?|"
                         r"kumquat|tray|container|table|surface)\b", re.IGNORECASE)


def split_fixtures(inventory: list[dict]) -> tuple[list[dict], list[dict]]:
    """(objects, dropped): inventory entries minus robot / fixture entries."""
    keep, drop = [], []
    for o in inventory:
        (drop if _FIXTURE_RE.search(f"{o.get('name', '')} {o.get('appearance', '')}")
         else keep).append(o)
    return keep, drop


# ---------------------------------------------------------------------------
# Live monitor: the offline run_scan decision flow, causal
# ---------------------------------------------------------------------------


class LiveMonitor:
    def __init__(self, client: TappedClient, model, processor, cfg: LiveMonitorConfig):
        self.client, self.model, self.processor, self.cfg = client, model, processor, cfg
        self.prompts = T.default_prompts()
        self.inbox: queue.Queue = queue.Queue()
        self._thread = None
        self._active = threading.Event()

    # -- per-episode state ----------------------------------------------------
    def start(self, sub_task: str, tag: str | None = None):
        self.sub_task = sub_task
        self.tag = tag                    # e.g. "trial_022": names the saved freeze frames
        self.front, self.wrist, self.joints, self.meta = [], [], [], []
        self.timeline = []
        self.reach_windows = 0            # two-pass reach checks run (2 VLM calls each)
        self.verify_calls = 0             # grounding / inventory / wrist-probe calls
        self.committed, self.held_object = False, None
        self.run_max_ap = None
        self.first_close = None
        self.gate_open_at = None          # close-trigger gate opening frame
        self.pending = None               # pre-grasp STOP awaiting the close
        self.verify_cache = self.grounded = self.inventory = None
        self.verify_info = {}
        self.stopped = None               # final STOP record
        self.grasp_decided = False        # gate=grasp: the single check has run
        self.inventory_dropped = []       # inventory entries dropped as robot / fixtures
        self.freeze_images = []           # saved grasp-freeze frames (paths)
        self.last_check_s = None          # wall time of the latest VLM check
        self.check_started_at = None      # episode time the latest check paused the arm
        self.stale_skipped = 0            # pre-pause frames that did not trigger a check
        self.inbox = queue.Queue()
        self.client.monitor_inbox = self.inbox
        self._active.set()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._active.clear()
        self._disarm_watch()
        self.client.monitor_inbox = None
        if self._thread is not None:
            self._thread.join(timeout=30)
        # Pending STOP never verified (episode ended first) -> it stands, unverified.
        if self.pending is not None and self.stopped is None:
            self.stopped = dict(self.pending, verify="unverified: episode ended before the close")
        self.client.execution_gate.set()

    # -- helpers ----------------------------------------------------------------
    def _source(self, lo: int = 0, hi: int | None = None) -> T.ClipSource:
        hi = len(self.front) if hi is None else hi
        paths = [f"{i:05d}_front_live.png" for i in range(lo, hi)]
        return T.ClipSource(self.front[lo:hi], paths, initial=self.front[0],
                            joints=self.joints[lo:hi], initial_joints=self.joints[0],
                            wrist=self.wrist[lo:hi])

    def _episode_elapsed(self) -> float | None:
        start = getattr(self.client, "_episode_start_perf", None)
        return None if start is None else time.perf_counter() - start

    def _arm_watch(self):
        """Arm the client's per-tick grasp watch for a pending STOP (live robot only)."""
        if not getattr(self.client, "supports_grasp_watch", False):
            return False
        g = [j[T._GRIPPER_IDX] for j in self.joints]
        with self.client._watch_lock:
            self.client.grasp_watch = {"start": g[0], "peak": max(g), "hist": [],
                                       "frozen_at": None, "frames": [], "ready": False}
        return True

    def _disarm_watch(self):
        if getattr(self.client, "supports_grasp_watch", False):
            with self.client._watch_lock:
                self.client.grasp_watch = None

    def _pause(self):
        self.client.execution_gate.clear()
        # Frames captured before this moment show the pre-pause scene: after the check
        # they may still arrive (a chunk binding during the pause), but must not trigger
        # another check of the same scene back-to-back (see _loop).
        self.check_started_at = self._episode_elapsed()

    def _resume(self):
        if self.stopped is None:
            self.client.execution_gate.set()

    # Timeline entries that ARE a VLM check (the others — COMMIT, RETIRED, FALLBACK — are
    # outcomes/notes of a check and must not read as extra checks).
    _CHECK_ACTIONS = {"CONTINUE", "STOP", "SUSPECT", "UNCLEAR", "VETOED", "PARSE_FAIL"}

    def _log(self, idx, action, reason, **extra):
        is_check = action in self._CHECK_ACTIONS
        rec = {"frame": idx, "t": round(self.meta[idx]["elapsed"], 2), "action": action,
               "reason": reason,
               "check_s": getattr(self, "last_check_s", None) if is_check else None, **extra}
        self.timeline.append(rec)
        took = f"(check {rec['check_s']}s) " if is_check else ""
        print(f"  [monitor] frame {idx:3d} t={rec['t']:5.1f}s  {action:9s} {took}{reason[:150]}",
              flush=True)

    def _do_stop(self, idx, reason, verify_note):
        self.stopped = {"frame": idx, "t": round(self.meta[idx]["elapsed"], 2),
                        "reason": reason, "verify": verify_note}
        self._log(idx, "STOP", f"[verify: {verify_note}] {reason}")
        self.client.episode_done.set()

    # -- main loop --------------------------------------------------------------
    def _loop(self):
        while self._active.is_set():
            w = getattr(self.client, "grasp_watch", None)
            if self.pending is not None and w is not None and w.get("ready") \
                    and self.stopped is None:
                try:
                    self._verify_and_decide(len(self.front) - 1, self.pending["reason"],
                                            fresh=w)
                except Exception as e:  # noqa: BLE001
                    logging.exception(f"[monitor] verify error: {e}")
                    self._resume()
                continue
            if (self.cfg.gate == "grasp" and not self.grasp_decided and self.pending is None
                    and w is not None and w.get("ready") and self.stopped is None):
                try:
                    self._single_check(len(self.front) - 1, fresh=w)
                except Exception as e:  # noqa: BLE001
                    logging.exception(f"[monitor] grasp check error: {e}")
                    self._resume()
                continue
            try:
                recs = [self.inbox.get(timeout=0.05 if self.pending else 0.2)]
            except queue.Empty:
                continue
            # Take everything that arrived meanwhile: all frames go into the history,
            # but only the NEWEST window is judged, so the monitor stays real-time
            # instead of working through a backlog of stale windows.
            while True:
                try:
                    recs.append(self.inbox.get_nowait())
                except queue.Empty:
                    break
            try:
                for rec in recs:
                    self._ingest(rec)
                # Stale-frame rule: frames captured before the last check started go into
                # the history (trajectory window, grasp-close detection) but do not
                # trigger a check — the next check waits for motion after the resume.
                fresh = (self.check_started_at is None
                         or recs[-1]["elapsed"] >= self.check_started_at)
                if self.stopped is None and self.front and fresh:
                    self._step(len(self.front) - 1)
                elif not fresh:
                    self.stale_skipped += len(recs)
            except Exception as e:  # noqa: BLE001 - a monitor error must never strand the arm
                logging.exception(f"[monitor] error: {e}")
                self._resume()

    def _ingest(self, rec):
        front = rec["frames"].get(self.cfg.vlm_camera_key)
        if front is None:
            return
        self.front.append(Image.fromarray(front))
        w = rec["frames"].get(self.cfg.wrist_key)
        if w is not None:
            im = Image.fromarray(w).convert("RGB")
            w = im.rotate(self.cfg.wrist_rotate, expand=True) if self.cfg.wrist_rotate % 360 else im
        self.wrist.append(w)
        self.joints.append(rec["state"])
        self.meta.append({"timestep": rec["timestep"], "elapsed": rec["elapsed"]})
        if self.first_close is None:
            c = T.first_grasp_close(self._source())
            if c is not None:
                self.first_close = c
                print(f"  [monitor] first grasp close observed at frame {c}", flush=True)

    def _step(self, idx: int):
        cfg = self.cfg
        # 1) A pre-grasp STOP waiting for the close: verify once close+1 has arrived.
        if self.pending is not None:
            if self.pending.get("watch"):
                pass  # the per-tick grasp watch freezes the arm and triggers the verify
            elif self.first_close is not None and idx >= self.first_close + 1:
                self._verify_and_decide(idx, self.pending["reason"],
                                        reach_unclear=self.pending.get("unclear", False))
            if self.stopped is None and self.pending is not None \
                    and idx - self.pending["frame"] >= cfg.pending_timeout_frames:
                self._do_stop(idx, self.pending["reason"],
                              f"no grasp close within {cfg.pending_timeout_frames} frames")
            return
        if (cfg.gate == "grasp" and not self.grasp_decided and not self.committed
                and getattr(self.client, "supports_grasp_watch", False)
                and getattr(self.client, "grasp_watch", None) is None):
            self._arm_watch()  # armed from the first frame so an early grasp is not missed
        if self.committed or len(self.front) < cfg.window:
            return
        if cfg.gate == "grasp":
            if self.grasp_decided:
                return
            if getattr(self.client, "supports_grasp_watch", False):
                # Live: the per-tick watch freezes the arm at the grasp and _loop runs the
                # single check; here only arm it and handle the no-grasp fallback.
                if getattr(self.client, "grasp_watch", None) is None:
                    self._arm_watch()
            elif self.first_close is not None:
                self._single_check(idx)  # replay: the window ending at the first close
                return
            g = [j[T._GRIPPER_IDX] for j in self.joints]
            opened = next((i for i, x in enumerate(g) if x - g[0] >= T.GRIPPER_OPEN_MIN), None)
            if opened is not None and idx - opened >= cfg.grasp_fallback_frames:
                self._log(idx, "FALLBACK", f"gripper opened @{opened} but no grasp within "
                          f"{cfg.grasp_fallback_frames} frames — checking once")
                self._single_check(idx)
            return
        src = self._source(idx - cfg.window + 1, idx + 1)
        cur = T.current_gripper(src.joints)
        if cur is not None:
            self.run_max_ap = cur if self.run_max_ap is None else max(self.run_max_ap, cur)
        closed = (cur <= self.run_max_ap - T.DEFAULT_GRASP_CLOSE_MARGIN
                  if cur is not None and self.run_max_ap is not None else None)
        gates = dict(motion_thresh=T.DEFAULT_MOTION_THRESH,
                     start_thresh=T.DEFAULT_JOINT_START_THRESH,
                     still_thresh=T.DEFAULT_JOINT_STILL_THRESH)
        # 2) Close-trigger gate (causal): open when the live queue plans the close.
        if cfg.gate == "close":
            if self.first_close is not None and idx > self.first_close + cfg.tail:
                return  # past the grasp: gate shut, check retired
            if self.gate_open_at is None:
                start_ap = self.joints[0][T._GRIPPER_IDX]
                peak = max(j[T._GRIPPER_IDX] for j in self.joints)
                ahead = self.client.queued_gripper(cfg.horizon)
                if ((peak - start_ap >= T.GRIPPER_OPEN_MIN and ahead and cur is not None
                        and cur - min(ahead) >= T.DEFAULT_GRASP_CLOSE_MARGIN)
                        or self.first_close is not None):
                    self.gate_open_at = idx
                    print(f"  [monitor] close-trigger gate OPEN at frame {idx}", flush=True)
                else:
                    return
            gates = dict(motion_thresh=0, start_thresh=0, still_thresh=0)
        # 3) Two-pass reach check (arm paused while the VLM runs).
        self._pause()
        try:
            target_object = self._ground() if cfg.ground_compare else None
            buf = io.StringIO()
            t_check = time.perf_counter()
            with contextlib.redirect_stdout(buf):
                verdict = T.run_check(self.model, self.processor, self.prompts, src,
                                      self.sub_task, cfg.temperature,
                                      gripper_closed=closed, target_object=target_object,
                                      **gates)
            self.last_check_s = round(time.perf_counter() - t_check, 2)
            gated = bool(verdict) and str(verdict.get("reaching_for", "")).startswith("none (arm not moving)")
            if not gated:
                self.reach_windows += 1
            action = "GATED" if gated else (str(verdict.get("action", "?")).upper() if verdict else "PARSE_FAIL")
            if action == "STOP" and cfg.verify:
                if self._arm_watch():
                    # Live: the per-tick watch freezes the arm at the grasp (or at once,
                    # if it has already closed) and verifies on fresh still frames.
                    self.pending = {"frame": idx, "t": round(self.meta[idx]["elapsed"], 2),
                                    "reason": verdict.get("reason", ""), "watch": True}
                    self._log(idx, "SUSPECT", "STOP held — arm will freeze the instant the "
                              f"gripper closes, then verify: {verdict.get('reason', '')}")
                    return
                if self.first_close is None:
                    self.pending = {"frame": idx, "t": round(self.meta[idx]["elapsed"], 2),
                                    "reason": verdict.get("reason", "")}
                    self._log(idx, "SUSPECT", "STOP raised before the grasp — holding it until "
                              f"the close to verify: {verdict.get('reason', '')}")
                    return
                if idx < self.first_close + 1:
                    self.pending = {"frame": idx, "t": round(self.meta[idx]["elapsed"], 2),
                                    "reason": verdict.get("reason", "")}
                    self._log(idx, "SUSPECT", "STOP at the close — waiting one frame to verify")
                    return
                self._verify_and_decide(idx, verdict.get("reason", ""))
                return
            if action == "STOP":
                self._do_stop(idx, verdict.get("reason", ""), "verify off")
                return
            if action != "GATED":
                self._log(idx, action, str(verdict.get("reason", "")) if verdict else "",
                          reaching_for=verdict.get("reaching_for") if verdict else None,
                          held=verdict.get("held") if verdict else None)
            if T._grasp_confirmed(verdict) and closed is not False:
                self.committed = True
                self.held_object = str(verdict.get("reaching_for") or "object")
                self._log(idx, "COMMIT", f"grasp confirmed on '{self.held_object}' — check retired")
        finally:
            self._resume()

    def _single_check(self, idx: int, fresh: "dict | None" = None):
        """gate=grasp: the one reach check of the run, on the look-back window ending at
        the latest frame, with the arm held; a STOP goes to the hybrid verify (on the
        watch's fresh still frames when given). Then the monitor retires."""
        cfg = self.cfg
        self.grasp_decided = True
        if fresh is not None:
            self._save_freeze(fresh)
        src = self._source(max(0, idx - cfg.window + 1), idx + 1)
        self._pause()
        try:
            target_object = self._ground() if cfg.ground_compare else None
            buf = io.StringIO()
            t_check = time.perf_counter()
            with contextlib.redirect_stdout(buf):
                verdict = T.run_check(self.model, self.processor, self.prompts, src,
                                      self.sub_task, cfg.temperature,
                                      gripper_closed=True if (fresh or self.first_close is not None) else None,
                                      target_object=target_object,
                                      motion_thresh=0, start_thresh=0, still_thresh=0)
            self.last_check_s = round(time.perf_counter() - t_check, 2)
            self.reach_windows += 1
            action = str(verdict.get("action", "?")).upper() if verdict else "PARSE_FAIL"
            reason = str(verdict.get("reason", "")) if verdict else ""
            if action == "STOP" and cfg.verify:
                if fresh is not None:
                    self._log(idx, "SUSPECT", f"single check says STOP — verifying on the "
                              f"frozen wrist: {reason}")
                    self._verify_and_decide(idx, reason, fresh=fresh)
                elif self.first_close is not None and idx >= self.first_close + 1:
                    self._verify_and_decide(idx, reason)
                else:
                    self.pending = {"frame": idx, "t": round(self.meta[idx]["elapsed"], 2),
                                    "reason": reason}
                    self._log(idx, "SUSPECT", f"single check says STOP — verifying at the "
                              f"close: {reason}")
                return
            if action == "STOP":
                self._do_stop(idx, reason, "verify off")
                return
            unclear = (str(verdict.get("reaching_for", "")).strip().lower()
                       in T._UNDECIDED_REACH) if verdict else False
            if (action == "CONTINUE" and unclear and cfg.verify
                    and getattr(cfg, "unclear_wrist_decides", False)
                    and (fresh is not None or self.first_close is not None)):
                self._log(idx, "UNCLEAR", f"single check unclear — the wrist decides: {reason}")
                if fresh is not None or idx >= self.first_close + 1:
                    self._verify_and_decide(idx, reason, fresh=fresh, reach_unclear=True)
                    if self.stopped is None:
                        self._log(idx, "RETIRED", "single grasp check done — monitor retired")
                        self._disarm_watch()
                else:
                    self.pending = {"frame": idx, "t": round(self.meta[idx]["elapsed"], 2),
                                    "reason": reason, "unclear": True}
                return
            self._log(idx, action, reason, reaching_for=verdict.get("reaching_for") if verdict else None,
                      held=verdict.get("held") if verdict else None)
            if T._grasp_confirmed(verdict):
                self.committed = True
                self.held_object = str(verdict.get("reaching_for") or "object")
                self._log(idx, "COMMIT", f"grasp confirmed on '{self.held_object}'")
            self._log(idx, "RETIRED", "single grasp check done — monitor retired for this run")
            self._disarm_watch()
        finally:
            self._resume()

    def _save_freeze(self, fresh: dict):
        """Write the grasp-freeze frames (front + wrist, wrist rotated as the VLM sees it)
        to <results_dir>/freeze/ for review. Never raises."""
        if self.freeze_images:
            return
        try:
            out = Path(self.cfg.results_dir) / "freeze"
            out.mkdir(parents=True, exist_ok=True)
            tag = self.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
            for i, f in enumerate(fresh.get("frames", [])):
                for cam, arr in f.get("frames", {}).items():
                    im = Image.fromarray(arr).convert("RGB")
                    if cam == self.cfg.wrist_key and self.cfg.wrist_rotate % 360:
                        im = im.rotate(self.cfg.wrist_rotate, expand=True)
                    path = out / f"{tag}_{cam}_{i}.png"
                    im.save(path)
                    self.freeze_images.append(str(path))
            print(f"  [monitor] saved {len(self.freeze_images)} freeze frames to {out}/",
                  flush=True)
        except Exception as e:  # noqa: BLE001 - review output must never break a run
            logging.warning(f"[monitor] could not save freeze frames: {e}")

    def _vocab(self) -> list[str]:
        if self.cfg.match == "inventory":
            if self.inventory is None:
                raw_inv = T.scene_inventory(self.model, self.processor, self.front[0],
                                            self.cfg.temperature)
                self.verify_calls += 1
                self.inventory, self.inventory_dropped = split_fixtures(raw_inv)
                print("  [monitor] scene inventory: " + "; ".join(
                    f"{o['name']} ({o['appearance']})" for o in self.inventory)
                    + (("  | dropped as robot/fixture: " + "; ".join(
                        o["name"] for o in self.inventory_dropped)) if self.inventory_dropped else ""),
                    flush=True)
            return [o["name"] for o in self.inventory]
        return list(self.cfg.objects)

    def _ground(self) -> str | None:
        """The sub-task's object grounded on the initial scene (once per episode, shared
        with the verify). Returns it only if it is one of the scene objects."""
        vocab = self._vocab()
        if self.grounded is None and vocab:
            raw = T.match_task_object(self.model, self.processor, self.front[0],
                                      self.sub_task, objects=vocab,
                                      temperature=self.cfg.temperature)["object"]
            self.verify_calls += 1
            self.grounded = raw if raw in ("none", "ambiguous") else (T.match_name(raw, vocab) or raw)
            print(f"  [monitor] sub-task object grounded on initial scene: '{self.grounded}'",
                  flush=True)
        return self.grounded if self.grounded in vocab else None

    def _verify_and_decide(self, idx: int, reach_reason: str, fresh: "dict | None" = None,
                           reach_unclear: bool = False):
        """Hybrid veto (the closed-list / inventory branch of run_scan): veto -> commit
        and continue; otherwise the STOP stands. With `fresh` (the grasp watch's two still
        frames taken while frozen at the grasp) the wrist probe runs on those; otherwise
        on the recorded close frame and the next one, as offline."""
        cfg, c = self.cfg, self.first_close
        self.pending = None
        self._pause()
        t_check = time.perf_counter()
        try:
            full = self._source()
            if fresh is not None:
                self._save_freeze(fresh)
                wr = []
                for f in fresh["frames"]:
                    im = f["frames"].get(cfg.wrist_key)
                    if im is not None:
                        im = Image.fromarray(im).convert("RGB")
                        im = im.rotate(cfg.wrist_rotate, expand=True) if cfg.wrist_rotate % 360 else im
                    wr.append(im)
                probe_src = T.ClipSource([full.initial] * len(wr),
                                         [f"fresh_{i}" for i in range(len(wr))],
                                         initial=full.initial, wrist=wr)
                c_desc = f"grasp freeze t={fresh['frozen_at']:.2f}s"
                self._disarm_watch()
            else:
                probe_src, c_desc = full, f"close@{c}"
            vocab = self._vocab()
            if self.verify_cache is None:
                T.HELD_PROBE_OBJECTS = list(cfg.objects)
                self.verify_cache = T.verify_held(
                    self.model, self.processor, probe_src,
                    0 if fresh is not None else c, cfg.temperature,
                    choices=self.inventory if cfg.match == "inventory" else None)
                self.verify_calls += 2
            held, answers = self.verify_cache
            seen = "/".join(a["object"] for a in answers)
            self._ground()
            self.last_check_s = round(time.perf_counter() - t_check, 2)
            self.verify_info = {"close_frame": c, "probe": c_desc,
                                "frozen_at": fresh["frozen_at"] if fresh else None,
                                "frozen_aperture": fresh.get("aperture") if fresh else None,
                                "held": held, "answers": answers,
                                "grounded": self.grounded, "inventory": self.inventory,
                                "verify_s": self.last_check_s}
            if reach_unclear:
                # The reach was unclear: only a clearly different object stops the arm.
                if held in vocab and self.grounded in vocab and held != self.grounded:
                    self._do_stop(idx, reach_reason, f"wrist-decided (reach unclear): wrist held "
                                  f"'{held}' ({seen}), task object '{self.grounded}'")
                else:
                    if held == self.grounded and self.grounded in vocab:
                        self.committed, self.held_object = True, held
                    self._log(idx, "CONTINUE", f"reach unclear; wrist '{held}' ({seen}) vs task "
                              f"object '{self.grounded}' — no STOP")
            elif held == "unclear":
                self._do_stop(idx, reach_reason, f"wrist blind or inconsistent at {c_desc} ({seen})")
            elif held == self.grounded and self.grounded in vocab:
                self.committed, self.held_object = True, held
                self._log(idx, "VETOED", f"wrist held '{held}' ({seen}) = task object "
                          f"'{self.grounded}' — STOP vetoed, grasp committed")
            else:
                self._do_stop(idx, reach_reason,
                              f"wrist held '{held}' ({seen}), task object grounded to '{self.grounded}'")
        finally:
            self._resume()


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

HELP = """Commands:
  /vla <task>        task sent to the VLA policy (what the arm does)
  /vlm <task>        sub-task the monitor believes (what it checks against)
  /both <task>       set both to the same task (a correct trial)
  /run  (or Enter)   run one episode (Enter during the episode aborts it)
  /gate grasp|close|distance   /horizon <steps>   /tail <frames>
  /unclear on|off    (grasp gate: the wrist decides when the look-back check is unclear)
  /verify on|off     /match scene|inventory   /objects a, b, c   /ground on|off
  /duration <s>      /home     /release (open gripper, then home)
  /config     /stats     /help     /quit
  (every episode ends with: open gripper -> home)"""


def main_repl(cfg: LiveMonitorConfig):
    logging.basicConfig(level=logging.INFO)
    results = Path(cfg.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    cfg.log_vla_inputs = True  # the monitor's frame tap relies on the input capture
    client = TappedClient(cfg)
    if not client.connect_server():
        client.logger.error("Could not connect to policy server. Exiting.")
        client.stop()
        return
    model, processor = T.load_model(cfg.vlm_model)
    monitor = LiveMonitor(client, model, processor, cfg)
    vla_task, vlm_task = cfg.vla_task, cfg.vlm_task
    tally = {"correct": [], "wrong": []}
    trial_n = len(list(results.glob("trial_*.json")))

    def show_config():
        grasp = (f"  [grasp: one check at the grasp freeze, unclear->wrist="
                 f"{'on' if cfg.unclear_wrist_decides else 'off'}, fallback "
                 f"{cfg.grasp_fallback_frames} frames]\n" if cfg.gate == "grasp" else "")
        print(f"  VLA task:     '{vla_task}'\n  monitor task: '{vlm_task}'\n" + grasp +
              f"  gate={cfg.gate} (horizon {cfg.horizon}, tail {cfg.tail})  verify={cfg.verify}  "
              f"match={cfg.match}  objects={cfg.objects}  ground_compare={cfg.ground_compare}  "
              f"duration={cfg.episode_duration}s")

    print("=" * 64 + "\nLIVE MONITOR TEST\n" + HELP + "\n" + "=" * 64)
    show_config()
    client.go_home()
    try:
        while True:
            try:
                line = input(f"\n[vla: '{vla_task}' | monitor: '{vlm_task}'] > ").strip()
            except EOFError:
                break
            cmd, _, arg = line.partition(" ")
            arg = arg.strip()
            if cmd in ("/quit", "/exit"):
                break
            elif cmd == "/help":
                print(HELP)
            elif cmd == "/config":
                show_config()
            elif cmd == "/vla" and arg:
                vla_task = arg
            elif cmd == "/vlm" and arg:
                vlm_task = arg
            elif cmd == "/both" and arg:
                vla_task = vlm_task = arg
            elif cmd == "/gate" and arg in ("distance", "close", "grasp"):
                cfg.gate = arg
            elif cmd == "/horizon" and arg.isdigit():
                cfg.horizon = int(arg)
            elif cmd == "/tail" and arg.isdigit():
                cfg.tail = int(arg)
            elif cmd == "/unclear" and arg in ("on", "off"):
                cfg.unclear_wrist_decides = arg == "on"
            elif cmd == "/verify" and arg in ("on", "off"):
                cfg.verify = arg == "on"
            elif cmd == "/ground" and arg in ("on", "off"):
                cfg.ground_compare = arg == "on"
            elif cmd == "/match" and arg in ("scene", "inventory"):
                cfg.match = arg
            elif cmd == "/objects" and arg:
                cfg.objects = [o.strip() for o in arg.split(",") if o.strip()]
            elif cmd == "/duration" and arg:
                cfg.episode_duration = float(arg)
            elif cmd == "/home":
                client.go_home()
            elif cmd == "/release":
                client.release_and_home()
            elif cmd == "/stats":
                for kind, want in (("correct", "CONTINUE"), ("wrong", "STOP")):
                    r = tally[kind]
                    print(f"  {kind:7} trials: {sum(x == want for x in r)}/{len(r)} as expected "
                          f"({want})")
            elif cmd in ("", "/run"):
                trial_n += 1
                print(f"\n--- trial {trial_n}: VLA '{vla_task}' | monitor '{vlm_task}' "
                      f"(Enter aborts) ---")
                monitor.start(vlm_task, tag=f"trial_{trial_n:03d}")
                t0 = time.time()
                result = client.run_episode(vla_task, enable_abort_listener=True)
                monitor.stop()
                outcome = "STOP" if monitor.stopped else "CONTINUE"
                print(f"\n=== MONITOR: {outcome}"
                      + (f" at frame {monitor.stopped['frame']} (t={monitor.stopped['t']}s): "
                         f"{monitor.stopped['verify']}" if monitor.stopped else
                         f" ({'committed to ' + repr(monitor.held_object) if monitor.committed else 'no decision'})")
                      + f" | reach windows {monitor.reach_windows}, verify calls "
                        f"{monitor.verify_calls} | {len(monitor.front)} frames ===")
                client.release_and_home()
                ans = input("Was the arm going for the MONITOR's object? "
                            "[y = correct trial / n = wrong trial / s = skip] ").strip().lower()
                label = {"y": "correct", "n": "wrong"}.get(ans[:1], None)
                if label:
                    tally[label].append(outcome)
                    ok = (outcome == "CONTINUE") == (label == "correct")
                    print(f"  -> {'OK' if ok else ('FALSE ALARM' if label == 'correct' else 'MISS')}")
                note = input("Note (optional): ").strip()
                out = {"trial": trial_n, "time": datetime.now().isoformat(timespec="seconds"),
                       "vla_task": vla_task, "monitor_task": vlm_task, "label": label,
                       "outcome": outcome, "stop": monitor.stopped,
                       "committed": monitor.committed, "held_object": monitor.held_object,
                       "reach_windows": monitor.reach_windows,
                       "stale_skipped": monitor.stale_skipped,
                       "verify_calls": monitor.verify_calls, "n_frames": len(monitor.front),
                       "first_close": monitor.first_close, "gate_open_at": monitor.gate_open_at,
                       "verify": monitor.verify_info, "timeline": monitor.timeline,
                       "freeze_images": monitor.freeze_images,
                       "inventory_dropped": monitor.inventory_dropped,
                       "episode": {"dir": str(client._vla_input_dir), **asdict(result)},
                       "wall_s": round(time.time() - t0, 1), "note": note,
                       "config": {k: getattr(cfg, k) for k in (
                           "gate", "horizon", "tail", "verify", "match", "objects", "window",
                           "ground_compare", "unclear_wrist_decides", "grasp_fallback_frames",
                           "freeze_stall_eps", "freeze_settle_s", "verify_frame_gap_s",
                           "pause_refreshes", "chunk_size_threshold", "pretrained_name_or_path",
                           "temperature", "episode_duration", "vlm_model")}}
                path = results / f"trial_{trial_n:03d}.json"
                path.write_text(json.dumps(out, indent=2, default=str))
                print(f"  saved {path} (episode frames: {client._vla_input_dir})")
            else:
                print("Unknown command or missing argument. /help for commands.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        client.release_and_home()
        client.stop()


@draccus.wrap()
def main(cfg: LiveMonitorConfig):
    main_repl(cfg)


if __name__ == "__main__":
    main()
