import torch
from torch import nn


class BlockWeightGenerator(nn.Module):
    """Generate matrix blocks from compact learned latent codes."""

    def __init__(
        self,
        block_count: int,
        block_size: int = 64,
        latent_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.block_size = block_size
        self.block_latent = nn.Embedding(block_count, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, block_size * block_size),
        )

    def forward(self, block_indices: torch.Tensor) -> torch.Tensor:
        """Return flattened generated blocks for matching block indices."""
        return self.decoder(self.block_latent(block_indices))
