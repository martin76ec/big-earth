from datasets import load_dataset

# Use streaming=True to explore without downloading 13GB+ immediately
ds = load_dataset("danielz01/BigEarthNet-S2-v1.0", split="train", streaming=True)

# Fetch the first sample
sample = next(iter(ds))
print(sample.keys())
