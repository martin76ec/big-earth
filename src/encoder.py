"""I-JEPA Encoder: ViT-Tiny backbone.

Standard Vision Transformer encoder with:
- Patchify + positional embeddings
- [CLS] token prepended to patch sequence
- 12 Transformer blocks (multi-head self-attention + MLP)
- LayerNorm on output tokens

Config: embed_dim=192, depth=12, heads=3, ~5M params.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.patchify import Patchify, NUM_PATCHES, IMG_SIZE


# ViT-Tiny config
EMBED_DIM = 192
DEPTH = 12
NUM_HEADS = 3
MLP_RATIO = 4  # hidden_dim = embed_dim * 4
DROPOUT = 0.0


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: LN → Attention → residual → LN → MLP → residual."""

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        mlp_ratio: int = MLP_RATIO,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        return x


class IJEPLEncoder(nn.Module):
    """ViT-Tiny encoder for I-JEPA.

    Parameters
    ----------
    img_size : int
        Input image spatial size (120).
    patch_size : int
        Patch size (10).
    in_channels : int
        Input channels (3 for RGB).
    embed_dim : int
        Token embedding dimension.
    depth : int
        Number of Transformer blocks.
    num_heads : int
        Number of attention heads.
    mlp_ratio : int
        MLP hidden dim = embed_dim * mlp_ratio.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        img_size: int = IMG_SIZE,
        patch_size: int = 10,
        in_channels: int = 3,
        embed_dim: int = EMBED_DIM,
        depth: int = DEPTH,
        num_heads: int = NUM_HEADS,
        mlp_ratio: int = MLP_RATIO,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Patchify (Conv2d projection)
        self.patchify = Patchify(embed_dim, patch_size, in_channels)

        # [CLS] token — learned global representation token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embeddings — one per patch + one for [CLS]
        num_positions = self.num_patches + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))

        # Transformer blocks
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )

        # Final norm
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        # Truncated normal for positional embeddings and CLS
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images into token representations.

        Args:
            x: [B, 3, 120, 120] image tensor

        Returns:
            dict with:
              - "tokens": [B, 145, embed_dim]  all tokens (CLS + 144 patches)
              - "cls":    [B, embed_dim]        CLS token only
        """
        B = x.shape[0]

        # Patchify: [B, 3, 120, 120] → [B, 144, embed_dim]
        patches = self.patchify(x)

        # Prepend [CLS]: [B, 145, embed_dim]
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)

        # Add positional embeddings
        tokens = tokens + self.pos_embed

        # Transformer blocks
        tokens = self.blocks(tokens)

        # Final norm
        tokens = self.norm(tokens)

        return {
            "tokens": tokens,       # [B, 145, embed_dim]
            "cls": tokens[:, 0],    # [B, embed_dim]
        }


if __name__ == "__main__":
    # Smoke test
    encoder = IJEPLEncoder()
    images = torch.randn(4, 3, 120, 120)
    out = encoder(images)

    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)

    print(f"Input:       {images.shape}")
    print(f"Tokens:      {out['tokens'].shape}")
    print(f"CLS:         {out['cls'].shape}")
    print(f"Num patches: {encoder.num_patches}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable:    {trainable_params:,}")