"""Replay recorded episodes through vlm_monitor_live.LiveMonitor (no robot) and print
its decision next to the offline hybrid result, to check the causal live port.

Run inside the lerobot-so101 container from /workspace:
    python vlm_monitor_test/live_replay_check.py [--gate close] [episodes...]
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, os.getcwd())
import vlm_monitor_tune as T  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from vlm_monitor_live import LiveMonitor  # noqa: E402
from vlm_monitor_test import WRONG_OBJECT, make_task, resolve_task  # noqa: E402


class FakeClient:
    """The slice of TappedClient the monitor uses; the queue look-ahead is the logged
    executed actions of the current frame's chunk."""

    def __init__(self):
        self.execution_gate = threading.Event()
        self.execution_gate.set()
        self.episode_done = threading.Event()
        self.monitor_inbox = None
        self.plan = []

    def queued_gripper(self, horizon):
        return self.plan[:horizon]


def load_episode(ep_dir):
    idx = lambda p: int(re.match(r"(\d+)", os.path.basename(p)).group(1))
    fronts = sorted(glob.glob(os.path.join(ep_dir, "*_front_*.png")), key=idx)
    wrists = {idx(p): p for p in glob.glob(os.path.join(ep_dir, "*_wrist_*.png"))}
    cols = ["in_shoulder_pan.pos", "in_shoulder_lift.pos", "in_elbow_flex.pos",
            "in_wrist_flex.pos", "in_wrist_roll.pos", "in_gripper.pos"]
    state, plan, elapsed = {}, defaultdict(list), {}
    for r in csv.DictReader(open(os.path.join(ep_dir, "actions.csv"))):
        i = idx(r["front_img"])
        state.setdefault(i, [float(r[c]) for c in cols])
        elapsed.setdefault(i, float(r["input_elapsed"]))
        plan[i].append((int(r["exec_index"]), float(r["act_gripper.pos"])))
    recs = []
    for p in fronts:
        i = idx(p)
        if i not in state:
            continue
        frames = {"front": np.asarray(Image.open(p).convert("RGB"))}
        if i in wrists:
            frames["wrist"] = np.asarray(Image.open(wrists[i]).convert("RGB"))
        recs.append(({"frames": frames, "state": state[i], "timestep": i,
                      "elapsed": elapsed[i]}, [g for _, g in sorted(plan[i])]))
    return recs


def replay(mon, client, recs, task):
    mon.start(task)
    for rec, plan in recs:
        if client.episode_done.is_set():
            break
        client.plan = plan
        mon.inbox.put(rec)
        # Wait until this frame is processed and the arm would be running again.
        while (not mon.inbox.empty() or not client.execution_gate.is_set()) \
                and not client.episode_done.is_set():
            time.sleep(0.05)
        time.sleep(0.3)
    mon.stop()
    return ("STOP" if mon.stopped else "CONTINUE"), mon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes", nargs="*", default=[
        "pouch/112_pouch", "roll/011_roll", "banana/001_banana", "brown/083_brown"])
    ap.add_argument("--gate", default="distance")
    ap.add_argument("--ground-compare", action="store_true")
    ap.add_argument("--unclear-wrist-decides", action="store_true")
    ap.add_argument("--match", default="scene", choices=["scene", "inventory"])
    ap.add_argument("--offline", default="vlm_monitor_test/hybrid_scene.json")
    args = ap.parse_args()
    offline = {f"{r['category']}/{r['episode']}": r
               for r in json.load(open(args.offline))["results"] if "correct" in r}
    # The monitor-side settings of LiveMonitorConfig (its robot/policy fields are not
    # needed without a robot).
    cfg = SimpleNamespace(
        vlm_model="Qwen/Qwen3-VL-8B-Instruct", temperature=0.1, window=4, gate=args.gate,
        horizon=T.DEFAULT_APPROACH_GATE_HORIZON, tail=1, verify=True, match=args.match,
        objects=list(T.HELD_PROBE_OBJECTS), vlm_camera_key="front", wrist_key="wrist",
        wrist_rotate=T.DEFAULT_WRIST_ROTATE_CCW, pending_timeout_frames=10,
        ground_compare=args.ground_compare,
        grasp_fallback_frames=T.DEFAULT_GRASP_FALLBACK_FRAMES,
        unclear_wrist_decides=args.unclear_wrist_decides)
    model, processor = T.load_model(cfg.vlm_model)
    for ep in args.episodes:
        ep_dir = os.path.join("vla_failure_test", ep)
        recs = load_episode(ep_dir)
        cat = ep.split("/")[0]
        src = T.load_clip(ep_dir)
        tasks = {"correct": resolve_task(src, cat),
                 "wrong": make_task(WRONG_OBJECT[offline[ep]["object_type"]])}
        for trial, task in tasks.items():
            client = FakeClient()
            mon = LiveMonitor(client, model, processor, cfg)
            live, m = replay(mon, client, recs, task)
            off = "STOP" if offline[ep][trial]["stopped"] else "CONTINUE"
            print(f"### {ep} {trial:7} '{task}': LIVE {live:8} | OFFLINE {off:8} | "
                  f"reach windows {m.reach_windows}, verify calls {m.verify_calls}, "
                  f"first close {m.first_close}, "
                  f"stop={m.stopped['verify'] if m.stopped else None}", flush=True)


if __name__ == "__main__":
    main()
