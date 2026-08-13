from lerobot.datasets.lerobot_dataset import LeRobotDataset

from collections import defaultdict

ds = LeRobotDataset("ant0nh/pnp_500_synonym_cleaned")

task_episodes = defaultdict(list)
for ep in range(ds.meta.total_episodes):
    task = str(ds.meta.episodes[ep]["tasks"])
    task_episodes[task].append(ep)

def to_ranges(eps):
    ranges = []
    start = eps[0]
    prev = eps[0]
    for e in eps[1:]:
        if e == prev + 1:
            prev = e
        else:
            ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
            start = e
            prev = e
    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ", ".join(ranges)

for task, eps in sorted(task_episodes.items()):
    print(f"{task} ({len(eps)} episodes): {to_ranges(eps)}")