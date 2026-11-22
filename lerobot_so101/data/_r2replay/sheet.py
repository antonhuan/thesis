import json, glob
from pathlib import Path
from PIL import Image, ImageDraw


def trial_path(n):
    """The replan-on original: archived when a replan-off re-run replaced it."""
    a = Path(f"preference_eval_system_trials/archive/trial_{n:03d}.json")
    return a if a.exists() else Path(f"preference_eval_system_trials/trial_{n:03d}.json")

cases = [12,14,16,17,20,24,28,29,34,39,40,43,44,45,46,47,50,53,55]
tiles = []
for n in cases:
    t = json.load(open(trial_path(n)))
    d = Path(t["system"]["task_dir"])
    f = sorted(d.glob("decompose_round2_*.png"))[0]
    im = Image.open(f).convert("RGB").resize((480, 360))
    ImageDraw.Draw(im).rectangle([0, 0, 480, 22], fill=(0, 0, 0))
    ImageDraw.Draw(im).text((4, 4), f"#{n} {t['prompt'][:60]}", fill=(255, 255, 0))
    tiles.append(im)
cols = 4; rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (cols * 480, rows * 360), (40, 40, 40))
for i, im in enumerate(tiles): sheet.paste(im, ((i % cols) * 480, (i // cols) * 360))
sheet.save("_r2replay/round2_frames.png")
