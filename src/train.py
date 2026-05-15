"""I-JEPA training script: pretrain → linear probe → evaluate.

Two phases:
  1) Self-supervised I-JEPA pretrain (no labels)
  2) Classifier linear probe (with labels, encoder frozen)

Usage:
  PYTHONPATH=. .venv/bin/python src/train.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import (
    BigEarthNetDataset,
    load_class_map,
    load_budgets,
    train_transform,
    eval_transform,
)
from src.encoder import IJEPLEncoder, EMBED_DIM
from src.predictor import IJEPAPredictor, MultiBlockMasking, ijepa_loss
from src.classifier import LinearProbe, build_loss, compute_pos_weights

# ── Config ──────────────────────────────────────────────

SEED = 42
BATCH_SIZE = 64
PRETRAIN_EPOCHS = 10
PROBE_EPOCHS = 20
LR_PRETRAIN = 1e-3
LR_PROBE = 1e-2
EVAL_EVERY = 1  # evaluate every N epochs

CHECKPOINT_DIR = Path("checkpoints")


# ── Helpers ─────────────────────────────────────────────

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)


def make_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders with reduced budget for speed."""
    class_map = load_class_map()
    budgets = load_budgets()

    # Use same reduced budget for all splits for fast iteration
    train_ds = BigEarthNetDataset("train", class_map, budgets=budgets, transform=train_transform)
    val_ds = BigEarthNetDataset("val", class_map, budgets=budgets, transform=eval_transform)
    test_ds = BigEarthNetDataset("test", class_map, budgets=budgets, transform=eval_transform)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_dl, val_dl, test_dl


# ── Phase 1: I-JEPA Pretrain ────────────────────────────

def pretrain(
    encoder: IJEPLEncoder,
    train_dl: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> None:
    """Self-supervised I-JEPA pretrain. No labels used."""
    predictor = IJEPAPredictor().to(device)
    masking = MultiBlockMasking()

    # Optimizer: encoder + predictor both updated
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr, weight_decay=0.05,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*60}", flush=True)
    print(f"PHASE 1: I-JEPA Pretrain ({epochs} epochs)")
    print(f"{'='*60}", flush=True)

    for epoch in range(epochs):
        encoder.train()
        predictor.train()
        epoch_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for images, _ in train_dl:
            images = images.to(device)
            B = images.shape[0]

            # Forward: encoder sees ALL patches
            encoder_out = encoder(images)
            encoder_tokens = encoder_out["tokens"]

            # Generate masks
            context_mask, target_mask = masking(B, device=device)

            # Predictor predicts target representations from context
            predictions, target_indices = predictor(
                encoder_tokens, context_mask, target_mask,
            )

            # Loss: MSE(predicted, stop-gradient(targets))
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
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"loss {avg_loss:.4f} | "
            f"lr {scheduler.get_last_lr()[0]:.6f} | "
            f"time {elapsed:.1f}s"
        )

    # Save pretrained encoder
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    enc_path = CHECKPOINT_DIR / "encoder_pretrained.pt"
    torch.save(encoder.state_dict(), enc_path)
    print(f"Saved encoder to {enc_path}", flush=True)


# ── Phase 2: Linear Probe ──────────────────────────────

def linear_probe(
    encoder: IJEPLEncoder,
    train_dl: DataLoader,
    val_dl: DataLoader,
    test_dl: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> None:
    """Train classifier head with frozen encoder."""
    class_map = load_class_map()
    budgets = load_budgets()

    # Freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # Head + loss
    head = LinearProbe().to(device)
    pos_weight = compute_pos_weights(class_map, budgets).to(device)
    criterion = build_loss(pos_weight)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    print(f"\n{'='*60}", flush=True)
    print(f"PHASE 2: Linear Probe ({epochs} epochs, encoder FROZEN)")
    print(f"{'='*60}", flush=True)

    for epoch in range(epochs):
        # ── Train ──
        head.train()
        train_loss = 0.0
        num_batches = 0

        for images, targets in train_dl:
            images, targets = images.to(device), targets.to(device)

            with torch.no_grad():
                cls = encoder(images)["cls"]  # [B, 192]

            logits = head(cls)  # [B, 5]
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / max(num_batches, 1)

        # ── Evaluate on val ──
        if (epoch + 1) % EVAL_EVERY == 0:
            val_metrics = evaluate(head, encoder, val_dl, device, "val")

            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"train_loss {avg_train_loss:.4f} | "
                f"val_loss {val_metrics['loss']:.4f} | "
                f"val_mAP {val_metrics['mAP']:.4f}"
            )

    # Final test evaluation
    test_metrics = evaluate(head, encoder, test_dl, device, "test")
    print(f"\n{'='*60}", flush=True)
    print(f"FINAL TEST RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Loss:  {test_metrics['loss']:.4f}", flush=True)
    print(f"mAP:   {test_metrics['mAP']:.4f}", flush=True)
    for name, idx in sorted(class_map.items(), key=lambda x: x[1]):
        ap = test_metrics["per_class_ap"].get(str(idx), 0.0)
        print(f"  {name}: AP={ap:.4f}", flush=True)

    # Save head
    head_path = CHECKPOINT_DIR / "head_linear.pt"
    torch.save(head.state_dict(), head_path)
    print(f"Saved head to {head_path}", flush=True)


# ── Evaluation ──────────────────────────────────────────

@torch.no_grad()
def evaluate(
    head: nn.Module,
    encoder: IJEPLEncoder,
    dataloader: DataLoader,
    device: torch.device,
    split_name: str = "val",
) -> dict:
    """Evaluate classifier: loss, mAP, per-class AP."""
    class_map = load_class_map()
    budgets = load_budgets()
    criterion = build_loss()  # no weights for fair eval

    all_probs = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0

    encoder.eval()
    head.eval()

    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)

        cls = encoder(images)["cls"]
        logits = head(cls)
        loss = criterion(logits, targets)

        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu())
        all_targets.append(targets.cpu())
        total_loss += loss.item()
        num_batches += 1

    all_probs = torch.cat(all_probs)     # [N, 5]
    all_targets = torch.cat(all_targets)  # [N, 5]

    # Per-class Average Precision
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
    """Compute Average Precision for a single class.

    Simplified implementation: sort by probability descending,
    compute precision-recall curve, integrate.
    """
    # Sort by probability descending
    sorted_indices = torch.argsort(probs, descending=True)
    sorted_targets = targets[sorted_indices]

    # Cumulative sums for precision-recall
    tp_cumsum = torch.cumsum(sorted_targets, dim=0)
    total_pos = targets.sum().item()

    if total_pos == 0:
        return 0.0

    # Precision at each threshold
    num_pred = torch.arange(1, len(sorted_targets) + 1, dtype=torch.float32)
    precision = tp_cumsum / num_pred

    # Average precision = mean of precisions at reciprocal ranks of positives
    recall_levels = tp_cumsum / total_pos
    ap = precision[sorted_targets == 1].mean().item() if (sorted_targets == 1).any() else 0.0

    return ap


# ── Main ────────────────────────────────────────────────

def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}", flush=True)
    print(f"Seed: {SEED}", flush=True)
    print(f"Batch size: {BATCH_SIZE}", flush=True)

    # Build dataloaders
    train_dl, val_dl, test_dl = make_dataloaders(BATCH_SIZE)

    # Build encoder
    encoder = IJEPLEncoder().to(device)

    # Phase 1: I-JEPA pretrain
    pretrain(encoder, train_dl, PRETRAIN_EPOCHS, LR_PRETRAIN, device)

    # Phase 2: Linear probe
    linear_probe(encoder, train_dl, val_dl, test_dl, PROBE_EPOCHS, LR_PROBE, device)

    print(f"\nDone. Checkpoints in {CHECKPOINT_DIR}/", flush=True)


if __name__ == "__main__":
    main()