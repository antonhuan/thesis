"""Recap grid of every recorded frame from one preference-eval trial.

Each tile is one VLA inference request: the front-camera frame, with the wrist-camera frame
inset at the bottom right and the elapsed time (from actions.csv) at the top left.
    python3 figures/trial_recap/make_trial_recap.py 36                 # baseline trial 36
    python3 figures/trial_recap/make_trial_recap.py 36 --n 20 --cols 5 # 20 evenly spaced frames
    python3 figures/trial_recap/make_trial_recap.py 36 --trials preference_eval_system_trials
Run from data/. Writes figures/trial_recap/recap_<trials dir>_trial<NNN>.png.
"""
import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

ap = argparse.ArgumentParser()
ap.add_argument("trial", type=int)
ap.add_argument("--trials", default="preference_eval_trials")
ap.add_argument("--cols", type=int, default=8)
ap.add_argument("--tile_w", type=int, default=480)    # front frames are 640x480
ap.add_argument("--no_wrist", action="store_true")
ap.add_argument("--title_size", type=int, default=40, help="task title font size (px)")
ap.add_argument("--no_info", action="store_true", help="title only: drop the trial info line")
ap.add_argument("--n", type=int, default=None,
                help="show only n evenly spaced frames (first and last always kept)")
args = ap.parse_args()

t = json.loads(Path(args.trials, f"trial_{args.trial:03d}.json").read_text())
ep = Path(t["episode"]["dir"])

# One row per frame: (elapsed seconds, front image, wrist image), in request order.
frames = {}
with open(ep / "actions.csv") as f:
    for r in csv.DictReader(f):
        frames.setdefault(r["front_img"], (float(r["input_elapsed"]), r["wrist_img"]))
frames = sorted(((e, fr, wr) for fr, (e, wr) in frames.items()), key=lambda x: x[0])
# Frames saved on disk but missing from actions.csv (e.g. the final request) go last, untimed.
listed = {fr for _, fr, _ in frames}
for p in sorted(ep.glob("*_front_*.png")):
    if p.name not in listed:
        frames.append((None, p.name, p.name.replace("_front_", "_wrist_")))
n_total = len(frames)
frames = [(k + 1, *fr) for k, fr in enumerate(frames)]      # keep the original frame number
if args.n and args.n < n_total:
    idx = sorted({round(i * (n_total - 1) / (args.n - 1)) for i in range(args.n)})
    frames = [frames[i] for i in idx]

tw = args.tile_w
th = tw * 3 // 4
gap, pad = 6, 24
cols = min(args.cols, len(frames))
rows = math.ceil(len(frames) / cols)
title_h = round(args.title_size * 1.3) + (14 if args.no_info else 58)
W = pad * 2 + cols * tw + (cols - 1) * gap
H = pad + title_h + rows * th + (rows - 1) * gap + pad
sheet = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(sheet)
f_title = ImageFont.truetype(FONT_BOLD, args.title_size)
f_sub = ImageFont.truetype(FONT, 28)
f_lab = ImageFont.truetype(FONT_BOLD, max(16, tw // 18))

placed = ", then ".join(t.get("placed") or []) or "nothing"
s = t["scores"]
outcome = "success" if s.get("complete") else ("violated" if s.get("violated") else "incomplete")
d.text((pad, pad), f'"{t["prompt"]}"', font=f_title, fill="#1a1a1a")
if not args.no_info:
    d.text((pad, pad + round(args.title_size * 1.3) + 2),
           f'Trial {t["trial"]} ({args.trials}) | category: {t["category"]} | layout: '
           f'{", ".join(f"{o} {sl}" for o, sl in zip(t["layout_order"], t["layout_slots"]))} | '
           f'placed: {placed} | {outcome} | {len(frames) if len(frames) == n_total else f'{len(frames)} of {n_total}'} frames over {t["episode"]["duration"]:.0f} s',
           font=f_sub, fill="#4a4a4a")

for i, (num, el, fr, wr) in enumerate(frames):
    x = pad + (i % cols) * (tw + gap)
    y = pad + title_h + (i // cols) * (th + gap)
    tile = Image.open(ep / fr).convert("RGB").resize((tw, th), Image.LANCZOS)
    if not args.no_wrist and (ep / wr).exists():
        iw = tw * 2 // 5
        ih = iw * 3 // 4
        inset = Image.open(ep / wr).convert("RGB").resize((iw, ih), Image.LANCZOS)
        ImageDraw.Draw(tile).rectangle([tw - iw - 6, th - ih - 6, tw - 2, th - 2], fill="white")
        tile.paste(inset, (tw - iw - 4, th - ih - 4))
    sheet.paste(tile, (x, y))
    lab = f"#{num}  t={el:.1f}s" if el is not None else f"#{num}"
    bx = d.textbbox((0, 0), lab, font=f_lab)
    d.rectangle([x, y, x + bx[2] + 14, y + bx[3] + 10], fill=(0, 0, 0))
    d.text((x + 7, y + 4), lab, font=f_lab, fill="white")

suffix = f"_n{len(frames)}" if len(frames) < n_total else ""
out = Path(__file__).parent / f"recap_{Path(args.trials).name}_trial{args.trial:03d}{suffix}.png"
sheet.save(out, optimize=True)
print(out, sheet.size)
