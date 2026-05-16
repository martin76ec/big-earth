"""I-JEPA training script: pretrain → linear probe → evaluate.

Two phases:
  1) Self-supervised I-JEPA pretrain (no labels)
  2) Classifier linear probe (with labels, encoder frozen)

Single GPU:
  PYTHONPATH=. python -u src/train.py

Multi-GPU (DDP):
  PYTHONPATH=. torchrun --nproc_per_node=8 -u src/train.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data import (
    BigEarthNetDataset,
    load_class_map,
    load_budgets,
    load_split,
    train_transform,
    eval_transform,
)
from src.encoder import IJEPLEncoder, EMBED_DIM
from src.predictor import IJEPAPredictor, MultiBlockMasking, ijepa_loss
from src.classifier import LinearProbe, MLPHead, build_loss, compute_pos_weights

# Suppress known driver-version warning on restricted servers.
warnings.filterwarnings(
    "ignore",
    message=".*The NVIDIA driver on your system is too old.*",
    category=UserWarning,
)

try:
    from sklearn.metrics import average_precision_score
except Exception:
    average_precision_score = None

# ── Config ──────────────────────────────────────────────

SEED = 42
BATCH_SIZE_PER_GPU = 64  # 64 for 8GB GPU, 256 for 80GB GPU (H200)
PRETRAIN_EPOCHS = 100
PROBE_EPOCHS = 30
LR_PRETRAIN = 1e-3
LR_PROBE = 1e-2
EVAL_EVERY = 1  # evaluate every N epochs
HEAD_TYPE = os.getenv("HEAD_TYPE", "mlp").lower()  # mlp | linear
DATA_MODE = os.getenv("DATA_MODE", "reduced").lower()  # reduced | full

CHECKPOINT_DIR = Path("checkpoints")


# ── DDP Helpers ─────────────────────────────────────────

def is_ddp() -> bool:
    """Check if running under torchrun (DDP mode)."""
    return "RANK" in os.environ


def get_rank() -> int:
    """Get current process rank (0 for single GPU or master)."""
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    """Get total number of GPUs (1 for single GPU)."""
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main_process() -> bool:
    """Only rank 0 prints and saves."""
    return get_rank() == 0


def setup_ddp() -> tuple[int, int, torch.device]:
    """Initialize DDP. Returns (rank, world_size, device)."""
    rank = get_rank()
    world_size = get_world_size()

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Synchronize all processes before starting
    dist.barrier()

    return rank, world_size, device


def cleanup_ddp() -> None:
    """Clean up DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ── Helpers ─────────────────────────────────────────────

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log(msg: str) -> None:
    """Print only from main process."""
    if is_main_process():
        print(msg, flush=True)


def make_dataloaders(
    batch_size: int,
    ddp: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test dataloaders.

    DATA_MODE=reduced -> 5% per-class budget (fast local iteration)
    DATA_MODE=full    -> full top-5 subset (slower, higher quality)
    """
    class_map = load_class_map()
    budgets = load_budgets() if DATA_MODE == "reduced" else None

    train_ds = BigEarthNetDataset("train", class_map, budgets=budgets, transform=train_transform)
    val_ds = BigEarthNetDataset("val", class_map, budgets=budgets, transform=eval_transform)
    test_ds = BigEarthNetDataset("test", class_map, budgets=budgets, transform=eval_transform)

    # Samplers: DistributedSampler for DDP, None for single GPU
    train_sampler = DistributedSampler(train_ds, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if ddp else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if ddp else None

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=(train_sampler is None),
        sampler=train_sampler, num_workers=2, pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        sampler=val_sampler, num_workers=2, pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        sampler=test_sampler, num_workers=2, pin_memory=True,
    )

    return train_dl, val_dl, test_dl, train_sampler


# ── Phase 1: I-JEPA Pretrain ────────────────────────────

def pretrain(
    encoder: IJEPLEncoder,
    train_dl: DataLoader,
    train_sampler: DistributedSampler | None,
    epochs: int,
    lr: float,
    device: torch.device,
    world_size: int,
) -> None:
    """Self-supervised I-JEPA pretrain. No labels used."""
    predictor = IJEPAPredictor().to(device)
    masking = MultiBlockMasking()

    # Wrap in DDP if multi-GPU
    if world_size > 1:
        encoder = DDP(encoder, device_ids=[device])
        predictor = DDP(predictor, device_ids=[device])

    # Effective learning rate scales with world size (linear scaling rule)
    effective_lr = lr * world_size

    # Optimizer: encoder + predictor both updated
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=effective_lr, weight_decay=0.05,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    log(f"\n{'='*60}")
    log(f"PHASE 1: I-JEPA Pretrain ({epochs} epochs, {world_size} GPU(s))")
    log(f"  Batch per GPU: {BATCH_SIZE_PER_GPU}")
    log(f"  Effective batch: {BATCH_SIZE_PER_GPU * world_size}")
    log(f"  LR: {effective_lr:.6f} (base {lr} × {world_size})")
    log(f"{'='*60}")

    for epoch in range(epochs):
        # Set epoch for DistributedSampler (ensures different shuffle per epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        encoder.train()
        predictor.train()
        epoch_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for images, _ in train_dl:
            images = images.to(device, non_blocking=True)
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

        log(
            f"Epoch {epoch+1:3d}/{epochs} | "
            f"loss {avg_loss:.4f} | "
            f"lr {scheduler.get_last_lr()[0]:.6f} | "
            f"time {elapsed:.1f}s"
        )

    # Save pretrained encoder (unwrap DDP, only main process)
    if is_main_process():
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        enc_state = encoder.module.state_dict() if world_size > 1 else encoder.state_dict()
        enc_path = CHECKPOINT_DIR / "encoder_pretrained.pt"
        torch.save(enc_state, enc_path)
        log(f"Saved encoder to {enc_path}")

    # Unwrap DDP for phase 2
    if world_size > 1:
        encoder = encoder.module


# ── Phase 2: Linear Probe ──────────────────────────────

def linear_probe(
    encoder: IJEPLEncoder,
    train_dl: DataLoader,
    train_sampler: DistributedSampler | None,
    val_dl: DataLoader,
    test_dl: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    world_size: int,
) -> None:
    """Train classifier head with frozen encoder."""
    class_map = load_class_map()
    budgets = load_budgets()

    # Freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    # Head + loss
    if HEAD_TYPE == "mlp":
        head = MLPHead().to(device)
    else:
        head = LinearProbe().to(device)
    if world_size > 1:
        head = DDP(head, device_ids=[device])

    pos_weight = compute_pos_weights(class_map, budgets).to(device)
    criterion = build_loss(pos_weight)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    log(f"\n{'='*60}")
    log(f"PHASE 2: Probe ({HEAD_TYPE.upper()} head, {epochs} epochs, encoder FROZEN, {world_size} GPU(s))")
    log(f"{'='*60}")

    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # ── Train ──
        head.train()
        train_loss = 0.0
        num_batches = 0

        for images, targets in train_dl:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

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
            val_metrics = evaluate(head, encoder, val_dl, device, world_size, "val")

            log(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"train_loss {avg_train_loss:.4f} | "
                f"val_loss {val_metrics['loss']:.4f} | "
                f"val_mAP {val_metrics['mAP']:.4f}"
            )

    # Unwrap DDP for eval
    if world_size > 1:
        head = head.module

    # Final test evaluation (only main process)
    if is_main_process():
        test_metrics = evaluate(head, encoder, test_dl, device, 1, "test")
        log(f"\n{'='*60}")
        log(f"FINAL TEST RESULTS")
        log(f"{'='*60}")
        log(f"Loss:  {test_metrics['loss']:.4f}")
        log(f"mAP:   {test_metrics['mAP']:.4f}")
        for name, idx in sorted(class_map.items(), key=lambda x: x[1]):
            ap = test_metrics["per_class_ap"].get(str(idx), 0.0)
            log(f"  {name}: AP={ap:.4f}")

        # Save head
        head_path = CHECKPOINT_DIR / "head_linear.pt"
        torch.save(head.state_dict(), head_path)
        log(f"Saved head to {head_path}")


# ── Evaluation ──────────────────────────────────────────

@torch.no_grad()
def evaluate(
    head: nn.Module,
    encoder: IJEPLEncoder,
    dataloader: DataLoader,
    device: torch.device,
    world_size: int,
    split_name: str = "val",
) -> dict:
    """Evaluate classifier: loss, mAP, per-class AP."""
    class_map = load_class_map()
    criterion = build_loss()  # no weights for fair eval

    all_probs = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0

    encoder.eval()
    head.eval()

    for images, targets in dataloader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        cls = encoder(images)["cls"]
        logits = head(cls)
        loss = criterion(logits, targets)

        probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu())
        all_targets.append(targets.cpu())
        total_loss += loss.item()
        num_batches += 1

    # Gather predictions from all GPUs if DDP
    if world_size > 1:
        gathered_probs = [torch.zeros_like(torch.cat(all_probs)) for _ in range(world_size)]
        gathered_targets = [torch.zeros_like(torch.cat(all_targets)) for _ in range(world_size)]
        dist.all_gather(gathered_probs, torch.cat(all_probs))
        dist.all_gather(gathered_targets, torch.cat(all_targets))
        all_probs = [p for g in gathered_probs for p in g.unsqueeze(0)]
        all_targets = [t for g in gathered_targets for t in t.unsqueeze(0)]

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
    """Compute Average Precision for a single class using sklearn metric."""
    y_true = targets.detach().cpu().numpy()
    y_score = probs.detach().cpu().numpy()
    if y_true.sum() == 0:
        return 0.0
    if average_precision_score is None:
        # Fallback (should rarely happen once sklearn is installed)
        sorted_indices = torch.argsort(probs, descending=True)
        sorted_targets = targets[sorted_indices]
        tp_cumsum = torch.cumsum(sorted_targets, dim=0)
        num_pred = torch.arange(1, len(sorted_targets) + 1, dtype=torch.float32)
        precision = tp_cumsum / num_pred
        return precision[sorted_targets == 1].mean().item() if (sorted_targets == 1).any() else 0.0
    return float(average_precision_score(y_true, y_score))


# ── Main ────────────────────────────────────────────────

def main() -> None:
    set_seed(SEED)

    # Detect DDP vs single GPU
    ddp_mode = is_ddp()

    if ddp_mode:
        rank, world_size, device = setup_ddp()
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Refusing to run to avoid accidental CPU training."
        )

    log(f"Device: {device}")
    log(f"Seed: {SEED}")
    log(f"Batch size/GPU: {BATCH_SIZE_PER_GPU}")
    log(f"Effective batch: {BATCH_SIZE_PER_GPU * world_size}")
    log(f"GPUs: {world_size}")
    log(f"DDP: {ddp_mode}")
    log(f"Head type: {HEAD_TYPE}")
    log(f"Data mode: {DATA_MODE}")

    # Prevent multi-rank HuggingFace cache races:
    # rank 0 pre-caches splits once, then all ranks proceed.
    if ddp_mode:
        if is_main_process():
            log("Pre-caching dataset splits on rank 0 before DDP loaders...")
            for split in ["train", "val", "test"]:
                load_split(split)
            log("Pre-cache complete.")
        dist.barrier()

    # Build dataloaders
    train_dl, val_dl, test_dl, train_sampler = make_dataloaders(
        BATCH_SIZE_PER_GPU, ddp=ddp_mode,
    )

    # Build encoder
    encoder = IJEPLEncoder().to(device)

    # Phase 1: I-JEPA pretrain
    pretrain(encoder, train_dl, train_sampler, PRETRAIN_EPOCHS, LR_PRETRAIN, device, world_size)

    # Phase 2: Linear probe
    linear_probe(encoder, train_dl, train_sampler, val_dl, test_dl, PROBE_EPOCHS, LR_PROBE, device, world_size)

    if is_main_process():
        log(f"\nDone. Checkpoints in {CHECKPOINT_DIR}/")

    # Clean up DDP
    if ddp_mode:
        cleanup_ddp()


if __name__ == "__main__":
    main()
