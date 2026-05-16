"""I-JEPA training script: pretrain → probe → evaluate (single GPU).

Usage:
  PYTHONPATH=. python -u src/train.py

Modes:
  DATA_MODE=reduced|full   (default: reduced)
  HEAD_TYPE=mlp|linear     (default: mlp)
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import BigEarthNetDataset, load_class_map, load_budgets, train_transform, eval_transform
from src.encoder import IJEPLEncoder
from src.predictor import IJEPAPredictor, MultiBlockMasking, ijepa_loss
from src.classifier import LinearProbe, MLPHead, build_loss, compute_pos_weights

try:
    from sklearn.metrics import average_precision_score
except Exception:
    average_precision_score = None

# Suppress known driver-version warning on restricted servers.
warnings.filterwarnings(
    "ignore",
    message=".*The NVIDIA driver on your system is too old.*",
    category=UserWarning,
)


SEED = 42
BATCH_SIZE = 64
PRETRAIN_EPOCHS = 100
PROBE_EPOCHS = 30
LR_PRETRAIN = 1e-3
LR_PROBE = 1e-2
EVAL_EVERY = 1

DATA_MODE = os.getenv("DATA_MODE", "reduced").lower()  # reduced | full
HEAD_TYPE = os.getenv("HEAD_TYPE", "mlp").lower()      # mlp | linear

CHECKPOINT_DIR = Path("checkpoints")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    class_map = load_class_map()
    budgets = load_budgets() if DATA_MODE == "reduced" else None

    train_ds = BigEarthNetDataset("train", class_map, budgets=budgets, transform=train_transform)
    val_ds = BigEarthNetDataset("val", class_map, budgets=budgets, transform=eval_transform)
    test_ds = BigEarthNetDataset("test", class_map, budgets=budgets, transform=eval_transform)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_dl, val_dl, test_dl


def pretrain(
    encoder: IJEPLEncoder,
    train_dl: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> None:
    predictor = IJEPAPredictor().to(device)
    masking = MultiBlockMasking()

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr,
        weight_decay=0.05,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*60}", flush=True)
    print(f"PHASE 1: I-JEPA Pretrain ({epochs} epochs)", flush=True)
    print(f"{'='*60}", flush=True)

    for epoch in range(epochs):
        encoder.train()
        predictor.train()
        epoch_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for images, _ in train_dl:
            images = images.to(device, non_blocking=True)
            B = images.shape[0]

            encoder_out = encoder(images)
            encoder_tokens = encoder_out["tokens"]

            context_mask, target_mask = masking(B, device=device)
            predictions, target_indices = predictor(encoder_tokens, context_mask, target_mask)
            loss = ijepa_loss(predictions, encoder_tokens, target_mask, target_indices)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1:3d}/{epochs} | loss {avg_loss:.4f} | "
            f"lr {scheduler.get_last_lr()[0]:.6f} | time {elapsed:.1f}s",
            flush=True,
        )

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    enc_path = CHECKPOINT_DIR / "encoder_pretrained.pt"
    torch.save(encoder.state_dict(), enc_path)
    print(f"Saved encoder to {enc_path}", flush=True)


def linear_probe(
    encoder: IJEPLEncoder,
    train_dl: DataLoader,
    val_dl: DataLoader,
    test_dl: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> None:
    class_map = load_class_map()
    budgets = load_budgets()

    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    head: nn.Module = MLPHead().to(device) if HEAD_TYPE == "mlp" else LinearProbe().to(device)
    pos_weight = compute_pos_weights(class_map, budgets).to(device)
    criterion = build_loss(pos_weight)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    print(f"\n{'='*60}", flush=True)
    print(f"PHASE 2: Probe ({HEAD_TYPE.upper()} head, {epochs} epochs, encoder FROZEN)", flush=True)
    print(f"{'='*60}", flush=True)

    for epoch in range(epochs):
        head.train()
        train_loss = 0.0
        num_batches = 0

        for images, targets in train_dl:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.no_grad():
                cls = encoder(images)["cls"]

            logits = head(cls)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / max(num_batches, 1)

        if (epoch + 1) % EVAL_EVERY == 0:
            val_metrics = evaluate(head, encoder, val_dl, device)
            print(
                f"Epoch {epoch+1:3d}/{epochs} | train_loss {avg_train_loss:.4f} | "
                f"val_loss {val_metrics['loss']:.4f} | val_mAP {val_metrics['mAP']:.4f}",
                flush=True,
            )

    test_metrics = evaluate(head, encoder, test_dl, device)
    print(f"\n{'='*60}", flush=True)
    print("FINAL TEST RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Loss:  {test_metrics['loss']:.4f}", flush=True)
    print(f"mAP:   {test_metrics['mAP']:.4f}", flush=True)
    for name, idx in sorted(class_map.items(), key=lambda x: x[1]):
        ap = test_metrics["per_class_ap"].get(str(idx), 0.0)
        print(f"  {name}: AP={ap:.4f}", flush=True)

    head_path = CHECKPOINT_DIR / "head_linear.pt"
    torch.save(head.state_dict(), head_path)
    print(f"Saved head to {head_path}", flush=True)


@torch.no_grad()
def evaluate(
    head: nn.Module,
    encoder: IJEPLEncoder,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    class_map = load_class_map()
    criterion = build_loss()

    all_probs = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0

    encoder.eval()
    head.eval()

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        cls = encoder(images)["cls"]
        logits = head(cls)
        loss = criterion(logits, targets)

        all_probs.append(torch.sigmoid(logits).cpu())
        all_targets.append(targets.cpu())
        total_loss += loss.item()
        num_batches += 1

    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)

    per_class_ap = {}
    aps = []
    for name, idx in class_map.items():
        ap = average_precision(all_targets[:, idx], all_probs[:, idx])
        per_class_ap[str(idx)] = ap
        per_class_ap[name] = ap
        aps.append(ap)

    mAP = sum(aps) / len(aps) if aps else 0.0
    return {
        "loss": total_loss / max(num_batches, 1),
        "mAP": mAP,
        "per_class_ap": per_class_ap,
    }


def average_precision(targets: torch.Tensor, probs: torch.Tensor) -> float:
    y_true = targets.detach().cpu().numpy()
    y_score = probs.detach().cpu().numpy()
    if y_true.sum() == 0:
        return 0.0
    if average_precision_score is None:
        sorted_indices = torch.argsort(probs, descending=True)
        sorted_targets = targets[sorted_indices]
        tp_cumsum = torch.cumsum(sorted_targets, dim=0)
        num_pred = torch.arange(1, len(sorted_targets) + 1, dtype=torch.float32)
        precision = tp_cumsum / num_pred
        return precision[sorted_targets == 1].mean().item() if (sorted_targets == 1).any() else 0.0
    return float(average_precision_score(y_true, y_score))


def main() -> None:
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Refusing to run to avoid accidental CPU training.")

    print(f"Device: {device}", flush=True)
    print(f"Seed: {SEED}", flush=True)
    print(f"Batch size: {BATCH_SIZE}", flush=True)
    print(f"Head type: {HEAD_TYPE}", flush=True)
    print(f"Data mode: {DATA_MODE}", flush=True)

    train_dl, val_dl, test_dl = make_dataloaders(BATCH_SIZE)
    encoder = IJEPLEncoder().to(device)

    pretrain(encoder, train_dl, PRETRAIN_EPOCHS, LR_PRETRAIN, device)
    linear_probe(encoder, train_dl, val_dl, test_dl, PROBE_EPOCHS, LR_PROBE, device)

    print(f"\nDone. Checkpoints in {CHECKPOINT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
