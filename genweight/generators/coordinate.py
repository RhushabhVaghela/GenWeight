import torch
from torch import nn


class CoordinateWeightGenerator(nn.Module):
    """Generate a matrix value from learned row and column embeddings."""

    def __init__(
        self,
        rows: int,
        columns: int,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.row_embedding = nn.Embedding(rows, embedding_dim)
        self.column_embedding = nn.Embedding(columns, embedding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self, row_indices: torch.Tensor, column_indices: torch.Tensor
    ) -> torch.Tensor:
        """Return generated weights for matching row and column index tensors."""
        coordinates = torch.cat(
            (self.row_embedding(row_indices), self.column_embedding(column_indices)),
            dim=-1,
        )
        return self.decoder(coordinates).squeeze(-1)
