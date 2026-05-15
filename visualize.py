import matplotlib.pyplot as plt
from datasets import load_dataset

# Use streaming=True to explore without downloading 13GB+ immediately
ds = load_dataset("danielz01/BigEarthNet-S2-v1.0", split="train", streaming=True)

# Fetch one sample
sample = next(iter(ds))

# Access the image and labels
image = sample['img']  # Note the key is 'img'
labels = sample['labels']

print(f"Labels: {labels}")
print(f"Image Type: {type(image)}")

# To view it
image.show()

# To convert to a NumPy array for analysis
import numpy as np
img_array = np.array(image)
print(f"Array Shape: {img_array.shape}")
