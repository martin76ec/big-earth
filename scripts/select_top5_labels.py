from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "danielz01/BigEarthNet-S2-v1.0"
SPLIT = "train"
TOP_K = 5
REDUCED_PERCENT = 0.05
FULL_OUTPUT_PATH = Path("labels.json")
REDUCED_OUTPUT_PATH = Path("labels_reduced.json")


def main() -> None:
    ds = load_dataset(DATASET_NAME, split=SPLIT, streaming=True)

    label_counts: Counter[str] = Counter()
    total_samples = 0
    total_labels = 0

    for sample in ds:
        labels = sample["labels"]
        label_counts.update(labels)
        total_samples += 1
        total_labels += len(labels)

    sorted_counts = sorted(label_counts.items(), key=lambda item: (-item[1], item[0]))
    top_k_counts = dict(sorted_counts[:TOP_K])

    reduced_counts = {
        label: max(1, int(count * REDUCED_PERCENT)) for label, count in top_k_counts.items()
    }

    FULL_OUTPUT_PATH.write_text(
        json.dumps(top_k_counts, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    REDUCED_OUTPUT_PATH.write_text(
        json.dumps(reduced_counts, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print(f"Dataset: {DATASET_NAME}")
    print(f"Split: {SPLIT}")
    print(f"Samples scanned: {total_samples}")
    print(f"Total labels scanned: {total_labels}")
    print(f"Saved top {TOP_K} labels to: {FULL_OUTPUT_PATH}")
    print(f"Saved reduced labels ({int(REDUCED_PERCENT * 100)}%) to: {REDUCED_OUTPUT_PATH}")
    print("Top labels:")
    for label, count in top_k_counts.items():
        print(f"- {label}: {count}")
    print("Reduced labels:")
    for label, count in reduced_counts.items():
        print(f"- {label}: {count}")


if __name__ == "__main__":
    main()
