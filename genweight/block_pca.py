from __future__ import annotations

import torch


class BlockPCABaseline:
    """Evaluate optimal linear low-rank reconstruction across matrix blocks."""

    def __init__(self, tensor: torch.Tensor, block_size: int = 64):
        if tensor.ndim != 2:
            raise ValueError("BlockPCABaseline requires a two-dimensional matrix.")
        if tensor.shape[0] % block_size or tensor.shape[1] % block_size:
            raise ValueError("matrix dimensions must be divisible by block_size.")

        self.tensor = tensor.detach().float().cpu()
        self.block_size = block_size
        self._blocks = self._extract_blocks()
        self._mean = self._blocks.mean(dim=0, keepdim=True)
        self._singular_values: torch.Tensor | None = None

    def _extract_blocks(self) -> torch.Tensor:
        rows, columns = self.tensor.shape
        row_blocks = rows // self.block_size
        column_blocks = columns // self.block_size
        blocks = self.tensor.reshape(
            row_blocks,
            self.block_size,
            column_blocks,
            self.block_size,
        )
        return blocks.permute(0, 2, 1, 3).reshape(-1, self.block_size**2)

    @property
    def singular_values(self) -> torch.Tensor:
        """Return singular values of the mean-centered block matrix."""
        if self._singular_values is None:
            self._singular_values = torch.linalg.svdvals(self._blocks - self._mean)
        return self._singular_values

    def evaluate(self, ranks: list[int]) -> list[dict[str, float | int]]:
        """Return exact PCA reconstruction and parameter storage estimates."""
        maximum_rank = min(self._blocks.shape)
        squared_singular_values = self.singular_values.square()
        total_squared_norm = self._blocks.square().sum()
        block_count, block_dimension = self._blocks.shape
        results = []

        for rank in sorted(set(ranks)):
            if not 1 <= rank <= maximum_rank:
                raise ValueError(f"rank must be between 1 and {maximum_rank}.")

            residual_squared_norm = squared_singular_values[rank:].sum()
            relative_error = torch.sqrt(residual_squared_norm / total_squared_norm).item()
            factor_parameters = block_dimension + rank * (block_count + block_dimension)
            results.append(
                {
                    "rank": rank,
                    "relative_frobenius_error": relative_error,
                    "factor_parameters": factor_parameters,
                    "parameter_ratio": factor_parameters / self.tensor.numel(),
                    "compression_ratio": self.tensor.numel() / factor_parameters,
                }
            )

        return results
