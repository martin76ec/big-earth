"""Classifier head for I-JEPA: linear probe + optional MLP.

Takes the CLS embedding from the I-JEPA encoder and maps it to
5 logits for multi-label classification.

Training uses BCEWithLogitsLoss (not CrossEntropy) because one
image can have multiple active classes.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.encoder import EMBED_DIM
from src.data import load_class_map

NUM_CLASSES = 5  # top-5 from labels.json


class LinearProbe(nn.Module):
    """Single linear layer on top of frozen encoder.

    This is the strongest baseline: if the encoder learned good
    representations, a linear classifier should already work well.
    If it doesn't, the encoder needs better pretraining — not a
    more complex head.

    Parameters
    ----------
    embed_dim : int
        CLS token dimension from encoder.
    num_classes : int
        Number of output classes (5).
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """Classify from CLS embedding.

        Args:
            cls_token: [B, embed_dim] from encoder

        Returns:
            logits: [B, num_classes] raw logits (NO sigmoid — loss handles that)
        """
        return self.head(cls_token)


class MLPHead(nn.Module):
    """Two-layer MLP head. Upgrade only if linear probe underfits.

    Parameters
    ----------
    embed_dim : int
        CLS token dimension from encoder.
    hidden_dim : int
        MLP hidden size.
    num_classes : int
        Number of output classes (5).
    dropout : float
        Dropout between layers.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = 96,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        return self.head(cls_token)


def build_loss(pos_weight: torch.Tensor | None = None) -> nn.BCEWithLogitsLoss:
    """Build multi-label classification loss.

    Args:
        pos_weight: per-class positive weight for imbalance.
            If None, all classes weighted equally.

    Returns:
        BCEWithLogitsLoss instance
    """
    if pos_weight is not None:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return nn.BCEWithLogitsLoss()


def compute_pos_weights(class_map: dict[str, int], budgets: dict[str, int]) -> torch.Tensor:
    """Compute per-class positive weights for imbalanced data.

    Weight = neg_count / pos_count for each class.
    This up-weights rare classes so the loss doesn't ignore them.

    Args:
        class_map: {class_name: index}
        budgets: {class_name: count} from labels_reduced.json

    Returns:
        pos_weight: [num_classes] tensor
    """
    total = sum(budgets.values())
    weights = torch.zeros(len(class_map))
    for name, idx in class_map.items():
        pos = budgets.get(name, 1)
        neg = total - pos
        weights[idx] = neg / pos
    return weights


if __name__ == "__main__":
    # Smoke test both heads
    class_map = load_class_map()

    cls_token = torch.randn(4, EMBED_DIM)  # 4 samples, 192-dim

    # Linear probe
    linear = LinearProbe()
    logits = linear(cls_token)
    print(f"=== Linear Probe ===")
    print(f"Input:  {cls_token.shape}")
    print(f"Output: {logits.shape}")
    print(f"Params: {sum(p.numel() for p in linear.parameters()):,}")

    # MLP head
    mlp = MLPHead()
    logits_mlp = mlp(cls_token)
    print(f"\n=== MLP Head ===")
    print(f"Input:  {cls_token.shape}")
    print(f"Output: {logits_mlp.shape}")
    print(f"Params: {sum(p.numel() for p in mlp.parameters()):,}")

    # Loss
    target = torch.tensor([
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1],
    ], dtype=torch.float32)

    loss = build_loss()(logits, target)
    print(f"\n=== Loss ===")
    print(f"Loss (no weights): {loss.item():.4f}")

    # With class weights
    from src.data import load_budgets
    budgets = load_budgets()
    pos_weight = compute_pos_weights(class_map, budgets)
    loss_weighted = build_loss(pos_weight)(logits, target)
    print(f"Loss (weighted):   {loss_weighted.item():.4f}")
    print(f"Pos weights:      {pos_weight.tolist()}")