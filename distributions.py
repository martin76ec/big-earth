from datasets import load_dataset
from collections import Counter

# Load in streaming mode to avoid a massive download
ds = load_dataset("danielz01/BigEarthNet-S2-v1.0", split="train", streaming=True)

label_counts = Counter()

print("Calculating distribution (this may take a few minutes)...")
# Iterate only over the labels to keep it fast
for sample in ds:
    label_counts.update(sample['labels'])

# Sort by frequency
sorted_counts = dict(sorted(label_counts.items(), key=lambda item: item[1], reverse=True))

for label, count in sorted_counts.items():
    print(f"{label}: {count}")
