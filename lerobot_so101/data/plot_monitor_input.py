"""Figure: what the live monitor's grasp check sees — a 2x4 grid (front row over wrist row,
oldest -> newest, left -> right) of its look-back window, plus one wrist still from the
grasp freeze on its own.

    python plot_monitor_input.py preference_eval_system_trials/runs/run_2026-10-10_06-58-38/02/monitor_round1_ep01_task01_attempt1.json

Writes monitor_input_grid.png and monitor_input_wrist.png next to the JSON's run folder
(or --out DIR). Wrist frames are rotated as the VLM sees them (MonitorSettings.wrist_rotate).

The window is frames idx-3..idx (idx = the check's frame in the monitor timeline). The
observation at idx — the one that triggered the grasp freeze — is never written to the
episode folder (its chunk id is always missing from actions.csv), so the newest column
uses the freeze still (front_0 / wrist_0, taken freeze_settle_s after the freeze with the
arm held at that pose).
"""
import argparse
import csv
import json
import os
import re

from PIL import Image

WINDOW = 4
WRIST_ROTATE_CCW = 90
GUTTER = 12


def episode_dir(monitor_json: str) -> str:
    log = open(os.path.join(os.path.dirname(monitor_json), "task.log")).read()
    eps = re.findall(r"VLA inputs for this episode -> (\S+)/", log)
    ep_n = int(re.search(r"_ep(\d+)_", os.path.basename(monitor_json)).group(1))
    return eps[ep_n - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("monitor_json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--wrist-anchor", type=float, default=0.5)
    args = ap.parse_args()

    d = json.load(open(args.monitor_json))
    idx = next(t["frame"] for t in d["timeline"] if t["action"] in ("CONTINUE", "STOP", "UNCLEAR"))
    ep = episode_dir(args.monitor_json)
    # chunk_id is 1-based: monitor frame i == chunk i + 1.
    by_chunk = {}
    for r in csv.DictReader(open(os.path.join(ep, "actions.csv"))):
        by_chunk.setdefault(int(r["chunk_id"]), (r["front_img"], r["wrist_img"]))

    freeze = {os.path.basename(p).rsplit("_", 2)[-2] + p[-6:-4]: p for p in d["freeze_images"]}
    front, wrist = [], []
    for i in range(idx - WINDOW + 1, idx + 1):
        if i + 1 in by_chunk:
            f, w = by_chunk[i + 1]
            front.append(Image.open(os.path.join(ep, f)).convert("RGB"))
            wrist.append(Image.open(os.path.join(ep, w)).convert("RGB").rotate(WRIST_ROTATE_CCW, expand=True))
        else:  # the freeze-trigger frame: use the freeze still (wrist already rotated)
            front.append(Image.open(freeze["front_0"]).convert("RGB"))
            wrist.append(Image.open(freeze["wrist_0"]).convert("RGB"))

    # Equal rows: wrist tiles scale to the front width and are cropped to the front's
    # aspect (the rotated wrist is portrait), at --wrist-anchor (0 top, 0.5 centre, 1 bottom).
    cw = front[0].width
    fh = front[0].height
    sh = round(wrist[0].height * cw / wrist[0].width)
    top = round((sh - fh) * args.wrist_anchor)
    wrist = [w.resize((cw, sh), Image.LANCZOS).crop((0, top, cw, top + fh)) for w in wrist]
    wh = fh
    grid = Image.new("RGB", (WINDOW * cw + (WINDOW - 1) * GUTTER, fh + GUTTER + wh), "white")
    for c in range(WINDOW):
        x = c * (cw + GUTTER)
        grid.paste(front[c], (x, 0))
        grid.paste(wrist[c], (x, fh + GUTTER))

    out = args.out or os.path.dirname(args.monitor_json)
    os.makedirs(out, exist_ok=True)
    grid.save(os.path.join(out, "monitor_input_grid.png"))
    Image.open(freeze["wrist_0"]).convert("RGB").save(os.path.join(out, "monitor_input_wrist.png"))
    print(f"{d['monitor_task']!r}: {d['outcome']}, held={d['held_object']}, window frames "
          f"{idx - WINDOW + 1}..{idx} of {ep} -> {out}")


if __name__ == "__main__":
    main()
