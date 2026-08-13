from lerobot.datasets.lerobot_dataset import LeRobotDataset
from collections import Counter

ds = LeRobotDataset("ant0nh/pnp_500_clean")

from collections import Counter

# Check what the tasks column looks like
print(ds.meta.episodes["tasks"][:5])

# If each entry is a list of task strings:
counts = Counter()
for task_list in ds.meta.episodes["tasks"]:
    for task in task_list:
        counts[task] += 1

for task, n in counts.items():
    print(f"{task}: {n} episodes")