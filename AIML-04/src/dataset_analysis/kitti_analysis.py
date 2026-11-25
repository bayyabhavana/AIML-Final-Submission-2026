from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt

LABEL_DIR = Path("datasets/kitti/training/label_2")
OUTPUT_DIR = Path("outputs/dataset_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

counts = Counter()
files = list(LABEL_DIR.glob("*.txt"))

if not files:
    print("No KITTI label files found.")
    raise SystemExit(1)

for file in files:
    with open(file, "r") as f:
        for line in f:
            values = line.split()
            if values:
                counts[values[0]] += 1

df = pd.DataFrame(
    counts.items(),
    columns=["Object Class", "Count"]
).sort_values("Count", ascending=False)

print("\nKITTI DATASET CLASS DISTRIBUTION")
print(df.to_string(index=False))

df.to_csv(OUTPUT_DIR / "kitti_class_distribution.csv", index=False)

plt.figure(figsize=(10, 6))
plt.bar(df["Object Class"], df["Count"])
plt.title("KITTI Object Class Distribution")
plt.xlabel("Object Class")
plt.ylabel("Number of Annotations")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "kitti_class_distribution.png")
plt.close()

print("\nTotal label files:", len(files))
print("Total annotations:", sum(counts.values()))
print("Results saved in:", OUTPUT_DIR)