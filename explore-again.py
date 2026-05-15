from datasets import load_dataset

ds = load_dataset("danielz01/BigEarthNet-S2-v1.0", split="train", streaming=True)
print(ds.features) # This will list all band names and label keys
