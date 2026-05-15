"""I-JEPA Predictor: context/target masking + prediction.

The predictor is the "personal trainer" that creates the learning
pressure during self-supervised pretrain. It takes context patch
representations + target position info, and predicts the encoder's
representations at target positions. After pretrain, this module
is DISCARDED.

Key I-JEPA insight: the encoder sees ALL patches. Masking only
applies to what the PREDICTOR gets as input. This is different from
MAE where the encoder only sees visible patches.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.patchify import NUM_PATCHES, IMG_SIZE
from src.encoder import EMBED_DIM

# Masking defaults
TARGET_SCALE = 0.30     # ~30% of patches are targets
CONTEXT_scale = 0.50    # remaining visible to predictor (rest are ignored by predictor)


class MultiBlockMasking(nn.Module):
    """Generate contiguous block masks for I-JEPA.

    Instead of random individual patches, we mask contiguous blocks.
    This forces the predictor to learn real spatial structure, not just
    copy neighbors.

    Strategy: pick random seed points, grow blocks around them until
    we fill the target budget.
    """

    def __init__(
        self,
        num_patches: int = NUM_PATCHES,
        grid_size: int = IMG_SIZE // 10,  # 12
        target_ratio: float = TARGET_SCALE,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.grid_size = grid_size
        self.target_ratio = target_ratio
        self.num_target = int(num_patches * target_ratio)

    def forward(self, B: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate masks for a batch.

        Args:
            B: batch size
            device: torch device

        Returns:
            context_mask: [B, num_patches] bool — True = visible to predictor
            target_mask:  [B, num_patches] bool — True = must be predicted
        """
        context_mask = torch.zeros(B, self.num_patches, dtype=torch.bool, device=device)
        target_mask = torch.zeros(B, self.num_patches, dtype=torch.bool, device=device)

        G = self.grid_size

        for b in range(B):
            # Pick random seed points for target blocks
            num_seeds = max(1, self.num_target // 6)  # ~6 patches per block on avg
            seeds = torch.randperm(self.num_patches, device=device)[:num_seeds]

            target_indices = set()
            for seed in seeds:
                # Convert flat index to grid coords
                r, c = seed.item() // G, seed.item() % G

                # Random block size around seed (1-3 in each direction)
                h = torch.randint(1, 4, (1,), device=device).item()
                w = torch.randint(1, 4, (1,), device=device).item()

                for dr in range(-h // 2, h // 2 + 1):
                    for dc in range(-w // 2, w // 2 + 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < G and 0 <= nc < G:
                            target_indices.add(nr * G + nc)

                if len(target_indices) >= self.num_target:
                    break

            # Trim to exact budget
            target_list = list(target_indices)[:self.num_target]
            target_mask[b, target_list] = True

            # Context = everything that's NOT target
            context_mask[b] = ~target_mask[b]

        return context_mask, target_mask


class IJEPAPredictor(nn.Module):
    """Lightweight predictor for I-JEPA pretrain.

    Takes context patch representations from the encoder + [MASK] tokens
    at target positions, and predicts the encoder's target representations.

    Architecture: shallow Transformer (2 blocks) — much lighter than encoder.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_patches: int = NUM_PATCHES,
        depth: int = 2,
        num_heads: int = 3,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches

        # Learnable [MASK] token — inserted at target positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings (same as encoder, shared not learned separately)
        # We'll receive pos info from the encoder, so we add our own here
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Shallow Transformer
        predictor_block = lambda: nn.ModuleDict({
            "norm1": nn.LayerNorm(embed_dim),
            "attn": nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, batch_first=True),
            "norm2": nn.LayerNorm(embed_dim),
            "mlp": nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim),
            ),
        })
        self.blocks = nn.ModuleList([predictor_block() for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(
        self,
        encoder_tokens: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict target representations from context.

        Args:
            encoder_tokens: [B, 145, embed_dim] — encoder output (CLS + patches)
            context_mask: [B, 144] bool — True = context patch
            target_mask: [B, 144] bool — True = target patch

        Returns:
            predictions: [B, num_target, embed_dim] — predicted target rep
            target_indices: [B, num_target] long — which patches were targeted
        """
        B = encoder_tokens.shape[0]

        # Extract patch tokens only (drop CLS at position 0)
        patch_tokens = encoder_tokens[:, 1:]  # [B, 144, embed_dim]

        # Build predictor input: context tokens at their positions,
        # [MASK] tokens at target positions
        pred_input = torch.empty_like(patch_tokens)  # [B, 144, embed_dim]

        # Place context tokens (with stop-gradient — predictor shouldn't
        # change the encoder through context, only the loss gradient does that)
        for b in range(B):
            pred_input[b, context_mask[b]] = patch_tokens[b, context_mask[b]].detach()
            pred_input[b, target_mask[b]] = self.mask_token.expand(1, target_mask[b].sum(), -1).squeeze(0)

        # Add positional embeddings (patch positions only, no CLS)
        pred_input = pred_input + self.pos_embed[:, 1:, :]

        # Run through predictor blocks (pre-norm)
        for block in self.blocks:
            h = block["norm1"](pred_input)
            h, _ = block["attn"](h, h, h, need_weights=False)
            pred_input = pred_input + h
            pred_input = pred_input + block["mlp"](block["norm2"](pred_input))

        pred_input = self.norm(pred_input)

        # Extract predictions at target positions only
        max_targets = target_mask.sum(dim=1).max().item()
        predictions = []
        target_indices = []
        for b in range(B):
            tidx = target_mask[b].nonzero(as_tuple=True)[0]
            predictions.append(pred_input[b, tidx])
            # Pad to max_targets for batching
            if len(tidx) < max_targets:
                pad = torch.zeros(max_targets - len(tidx), pred_input.shape[-1], device=pred_input.device)
                predictions[-1] = torch.cat([predictions[-1], pad])
                tidx = torch.cat([tidx, torch.zeros(max_targets - len(tidx), dtype=torch.long, device=pred_input.device)])
            target_indices.append(tidx)

        predictions = torch.stack(predictions)       # [B, max_targets, embed_dim]
        target_indices = torch.stack(target_indices) # [B, max_targets]

        return predictions, target_indices


def ijepa_loss(
    predictions: torch.Tensor,
    encoder_tokens: torch.Tensor,
    target_mask: torch.Tensor,
    target_indices: torch.Tensor,
) -> torch.Tensor:
    """I-JEPA loss: MSE between predicted and stop-gradient target representations.

    Gradients flow through: predictor + encoder (via context path).
    Gradients blocked at: target representations (stop-gradient).
    """
    B = predictions.shape[0]
    patch_tokens = encoder_tokens[:, 1:]  # [B, 144, embed_dim]

    total_loss = torch.tensor(0.0, device=predictions.device)
    total_count = 0

    for b in range(B):
        tidx = target_mask[b].nonzero(as_tuple=True)[0]
        # Target representations with stop-gradient
        targets = patch_tokens[b, tidx].detach()
        # Predictions for those positions
        preds = predictions[b, :len(tidx)]
        total_loss = total_loss + torch.nn.functional.mse_loss(preds, targets)
        total_count += 1

    return total_loss / total_count


if __name__ == "__main__":
    from src.encoder import IJEPLEncoder

    # Full I-JEPA forward pass smoke test
    encoder = IJEPLEncoder()
    predictor = IJEPAPredictor()
    masking = MultiBlockMasking()

    images = torch.randn(2, 3, 120, 120)
    B = images.shape[0]

    # Step 1: Encode full image
    encoder_out = encoder(images)
    encoder_tokens = encoder_out["tokens"]  # [B, 145, 192]

    # Step 2: Generate masks
    context_mask, target_mask = masking(B, device=images.device)

    # Step 3: Predict targets from context
    predictions, target_indices = predictor(encoder_tokens, context_mask, target_mask)

    # Step 4: Compute loss
    loss = ijepa_loss(predictions, encoder_tokens, target_mask, target_indices)

    enc_params = sum(p.numel() for p in encoder.parameters())
    pred_params = sum(p.numel() for p in predictor.parameters())

    print(f"=== I-JEPA Full Forward Pass ===")
    print(f"Images:          {images.shape}")
    print(f"Encoder tokens:  {encoder_tokens.shape}")
    print(f"Context patches: {context_mask[0].sum().item()} / {NUM_PATCHES}")
    print(f"Target patches:  {target_mask[0].sum().item()} / {NUM_PATCHES}")
    print(f"Predictions:     {predictions.shape}")
    print(f"Loss:            {loss.item():.6f}")
    print(f"")
    print(f"Encoder params:  {enc_params:,}")
    print(f"Predictor params: {pred_params:,}")
    print(f"Total:           {enc_params + pred_params:,}")