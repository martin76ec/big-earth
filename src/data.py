"""BigEarthNet data pipeline for I-JEPA classifier.

Downloads the dataset once to local disk (if not cached), then loads
from disk for fast reliable training. Filters to top-5 classes, respects
per-class budgets from labels_reduced.json, and yields
(image_tensor, multi_hot_target) pairs.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import os

import torch
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar, set_verbosity_error
from torch.utils.data import Dataset
from torchvision.transforms import v2

DATASET_NAME = "danielz01/BigEarthNet-S2-v1.0"
CACHE_DIR = Path("data/bigearthnet")
LABELS_PATH = Path("labels.json")
LABELS_REDUCED_PATH = Path("labels_reduced.json")

# ImageNet normalization (standard for ViT-style models)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# Train augmentations
train_transform = v2.Compose(
    [
        v2.RandomResizedCrop(size=(120, 120), scale=(0.8, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEAN, std=STD),
    ]
)

# Eval augmentations (deterministic, no randomness)
eval_transform = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEAN, std=STD),
    ]
)


def _is_main_process() -> bool:
    """Only rank 0 should print in DDP."""
    return int(os.environ.get("RANK", "0")) == 0


def _log(msg: str) -> None:
    if _is_main_process():
        print(msg, flush=True)


def _configure_hf_logging_for_rank() -> None:
    """Silence HF datasets progress/logging for non-main DDP ranks."""
    if not _is_main_process():
        disable_progress_bar()
        set_verbosity_error()


def load_split(split: str):
    """Load a dataset split, downloading to cache on first call.

    Uses HuggingFace's built-in caching — downloads once, loads from
    local Arrow files thereafter. No manual save_to_disk needed.
    """
    _configure_hf_logging_for_rank()
    _log(f"  Loading {split} split ...")
    ds = load_dataset(DATASET_NAME, split=split, cache_dir=str(CACHE_DIR))
    _log(f"  {split}: {len(ds)} samples loaded")
    return ds


def load_class_map(labels_path: Path = LABELS_PATH) -> dict[str, int]:
    """Load class names and produce a deterministic class→index map.

    Order follows the JSON key order (which is count-desc from the
    generator script).
    """
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    return {name: idx for idx, name in enumerate(labels.keys())}


def load_budgets(reduced_path: Path = LABELS_REDUCED_PATH) -> dict[str, int]:
    """Load per-class sample budgets from the reduced labels file."""
    return json.loads(reduced_path.read_text(encoding="utf-8"))


def labels_to_multi_hot(
    label_names: list[str], class_map: dict[str, int]
) -> torch.Tensor:
    """Convert a list of label names to a multi-hot tensor."""
    target = torch.zeros(len(class_map), dtype=torch.float32)
    for name in label_names:
        if name in class_map:
            target[class_map[name]] = 1.0
    return target


class BigEarthNetDataset(Dataset):
    """Filtered BigEarthNet dataset yielding (image_tensor, multi_hot_target).

    Loads from local disk after initial download. Filters to top-5 classes
    and respects per-class budgets.

    Parameters
    ----------
    split : str
        HuggingFace dataset split ("train", "val", "test").
    class_map : dict[str, int]
        Mapping from class name to index.
    budgets : dict[str, int] | None
        Per-class sample caps. None = no budget (use full split).
    transform : torchvision transform or None
        Applied to each image after loading.
    """

    def __init__(
        self,
        split: str,
        class_map: dict[str, int],
        budgets: dict[str, int] | None = None,
        transform=None,
    ) -> None:
        self.class_map = class_map
        self.transform = transform
        self.split = split

        # Load dataset (downloads on first call, cached thereafter)
        ds = load_split(split)

        spent = Counter[str]()
        indices = []  # store indices, not raw samples (saves memory)
        scanned = 0

        for i, sample in enumerate(ds):
            scanned += 1
            if scanned % 50000 == 0:
                _log(f"  scanned {scanned:,}, kept {len(indices):,} ...")

            label_names = sample["labels"]
            top_labels = [l for l in label_names if l in class_map]

            # Skip samples with no top-5 labels
            if not top_labels:
                continue

            # If budgets active, check each class still has quota
            if budgets is not None:
                over_budget = [l for l in top_labels if spent[l] >= budgets.get(l, 0)]
                if over_budget:
                    continue
                for l in top_labels:
                    spent[l] += 1

                # Early exit: all budgets filled
                if all(spent[l] >= budgets.get(l, 0) for l in class_map):
                    indices.append(i)
                    break

            indices.append(i)

        self.ds = ds
        self.indices = indices
        total_samples = len(indices)
        total_budget = sum(budgets.values()) if budgets else "unlimited"
        _log(f"Kept {total_samples} samples (budget: {total_budget})")
        if budgets is not None:
            for name, idx in sorted(class_map.items(), key=lambda x: x[1]):
                cap = budgets.get(name, "∞") if budgets else "∞"
                used = spent.get(name, 0)
                _log(f"  class {idx}: {name} → {used}/{cap}")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.indices[idx]
        sample = self.ds[real_idx]
        image = sample["img"]
        label_names = sample["labels"]

        if self.transform:
            image = self.transform(image)

        target = labels_to_multi_hot(label_names, self.class_map)
        return image, target


if __name__ == "__main__":
    # Quick smoke test with reduced budget
    class_map = load_class_map()
    budgets = load_budgets()

    ds = BigEarthNetDataset(
        split="train",
        class_map=class_map,
        budgets=budgets,
        transform=train_transform,
    )

    img, target = ds[0]
    print(f"\nSample 0:")
    print(f"  image shape: {img.shape}")
    print(f"  target:      {target}")
    print(f"  target len:  {len(target)}")
