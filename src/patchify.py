"""Patchify module for I-JEPA.

Converts a 120×120×3 image into a sequence of patch tokens
using patch_size=10 → 12×12 grid = 144 patches.

Uses Conv2d projection (the standard ViT approach): a single
convolution with kernel_size=stride=patch_size is equivalent to
"cut into patches and project each one" but runs as one高效 op.
"""

from __future__ import annotations

import torch
import torch.nn as nn

PATCH_SIZE = 10
IMG_SIZE = 120
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2  # 144


class Patchify(nn.Module):
    """Convert images into patch token sequences.

    Parameters
    ----------
    embed_dim : int
        Output embedding dimension per patch.
    patch_size : int
        Spatial size of each square patch (10 for our config).
    in_channels : int
        Number of input image channels (3 for RGB).
    """

    def __init__(
        self,
        embed_dim: int = 192,
        patch_size: int = PATCH_SIZE,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (IMG_SIZE // patch_size) ** 2

        # Conv2d with kernel=stride=patch_size does the cutting AND
        # projection in one step. Each output channel = one embed dim.
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify a batch of images.

        Args:
            x: [B, 3, 120, 120] image tensor

        Returns:
            [B, 144, embed_dim] patch token sequence
        """
        # [B, embed_dim, H/P, W/P] → [B, embed_dim, num_patches] → [B, num_patches, embed_dim]
        tokens = self.proj(x).flatten(2).transpose(1, 2)
        return tokens


if __name__ == "__main__":
    # Smoke test
    patchify = Patchify(embed_dim=192)

    # Simulate a batch of 4 images
    images = torch.randn(4, 3, 120, 120)
    tokens = patchify(images)

    print(f"Input:  {images.shape}")
    print(f"Output: {tokens.shape}")
    print(f"Expected: [4, 144, 192]")
    print(f"Num patches: {patchify.num_patches}")
    print(f"Proj params: {sum(p.numel() for p in patchify.proj.parameters()):,}")