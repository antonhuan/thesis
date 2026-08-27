import re, csv, os

base = os.path.dirname(os.path.abspath(__file__))
objects = ["banana", "pouch", "toy"]
metrics = ["Mean Std", "Max Std", "Mean CV", "Max CV", "Action Mag",
           "EE Consist", "Peak Disp", "No-Move %", "GT Dist"]

rows = []
for obj in objects:
    path = os.path.join(base, obj, "denoise_results.md")
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not re.match(r'^[\s*]', line) or line.startswith('--') or line.startswith('Label'):
                continue
            train = line.startswith('*')
            parts = re.split(r'\s{2,}', line.strip().lstrip('*').strip())
            if len(parts) < 11:
                continue
            label = parts[0]
            prompt = parts[1]
            for i, m in enumerate(metrics):
                val_str = parts[i + 2]
                match = re.match(r'([\d.]+)\+-([\d.]+)', val_str)
                if match:
                    mean, sd = float(match.group(1)), float(match.group(2))
                    rows.append([obj, label, "yes" if train else "no",
                                 prompt, m, mean, sd])

out = os.path.join(base, "denoise_results_all.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["object", "label", "training_label", "prompt", "metric", "mean", "sd"])
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to {out}")
