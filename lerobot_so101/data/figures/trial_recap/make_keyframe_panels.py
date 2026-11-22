"""Key-frame panels for single baseline (VLA only) trials, in the style of
figures/poster_examples/make_panels.py: prompt on top, one grey "Baseline (VLA only)" row of
front-camera frames, each captioned above. Frame numbers are the episode's file numbers
(NNNNN_front_*.png); -1 means the last frame.
    python3 figures/trial_recap/make_keyframe_panels.py          # all CASES
    python3 figures/trial_recap/make_keyframe_panels.py omit_pouch
Run from data/. Writes figures/trial_recap/<name>_panel.png.
"""
import glob
import json
import sys

from PIL import Image, ImageDraw, ImageFont

OUT = "figures/trial_recap"
FB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 42)  # prompt
FS = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)       # captions
FR = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)  # row heading
COL, TCOL = "#a3a3a3", "#6b6b6b"   # baseline grey, as in plot_pref_category.py / make_panels.py

# name: (baseline trial, [(frame, caption)], row label)
CASES = {
    "omit_pouch": (2, [(0, "start"), (20, "toy placed"), (34, "grasps the pouch"),
                       (-1, "pouch placed: violated")], "Baseline (VLA only)"),
    "cond_brown": (43, [(0, "start"), (16, "toy placed"), (30, "grasps the pouch"),
                        (-1, "both placed: correct")], "Baseline (VLA only)"),
}

W, H = 440, 330; PAD = 10; BW = 5          # frame size, gutter, border width
TOP = 70; HEAD = 50; LINE = 44             # prompt band, row heading, caption line height


def wrap(D, text, font, width):
    lines = [""]
    for w in text.split(" "):
        t = (lines[-1] + " " + w).strip()
        if D.textlength(t, font=font) <= width - 8:
            lines[-1] = t
        else:
            lines.append(w)
    return lines


def frame(d, n):
    fronts = sorted(glob.glob(f"{d}/*_front_*.png"))
    return fronts[n] if n < 0 else glob.glob(f"{d}/{n:05d}_front_*.png")[0]


for name in sys.argv[1:] or CASES:
    trial, frames, label = CASES[name]
    t = json.load(open(f"preference_eval_trials/trial_{trial:03d}.json"))
    d = t["episode"]["dir"]
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    cap_h = max(len(wrap(probe, c, FS, W)) for _, c in frames) * LINE + 6
    n = len(frames)
    P = Image.new("RGB", (n * W + (n + 1) * PAD, TOP + HEAD + cap_h + H + PAD), "white")
    D = ImageDraw.Draw(P)
    D.text((PAD, 12), f'"{t["prompt"]}"', fill="black", font=FB)
    D.text((PAD, TOP + 4), label, fill=TCOL, font=FR)
    for c, (fn, cap) in enumerate(frames):
        x = PAD + c * (W + PAD)
        for i, ln in enumerate(wrap(D, cap, FS, W)):
            D.text((x + 2, TOP + HEAD + i * LINE), ln, fill=TCOL, font=FS)
        yi = TOP + HEAD + cap_h
        P.paste(Image.open(frame(d, fn)).convert("RGB").resize((W, H), Image.LANCZOS), (x, yi))
        D.rectangle([x, yi, x + W - 1, yi + H - 1], outline=COL, width=BW)
    P.save(f"{OUT}/{name}_panel.png")
    print(name, f"trial {trial}", d, P.size)
