from __future__ import annotations

from itertools import combinations

import torch


class QKVSimilarityAnalyzer:
    """Compare aligned Query, Key, and Value blocks in a fused attention matrix."""

    def __init__(self, tensor: torch.Tensor, block_size: int = 64):
        if tensor.ndim != 2:
            raise ValueError("QKVSimilarityAnalyzer requires a two-dimensional matrix.")
        if tensor.shape[1] % 3:
            raise ValueError("the fused attention matrix must have three equal column segments.")
        if tensor.shape[0] % block_size or (tensor.shape[1] // 3) % block_size:
            raise ValueError("matrix dimensions must be divisible by block_size.")

        self.tensor = tensor.detach().float().cpu()
        self.block_size = block_size

    def _segment_blocks(self) -> torch.Tensor:
        """Return blocks shaped as [QKV segment, block index, flattened values]."""
        rows, columns = self.tensor.shape
        segment_width = columns // 3
        row_blocks = rows // self.block_size
        column_blocks = segment_width // self.block_size
        segments = self.tensor.reshape(rows, 3, segment_width).permute(1, 0, 2)
        blocks = segments.reshape(
            3,
            row_blocks,
            self.block_size,
            column_blocks,
            self.block_size,
        )
        return blocks.permute(0, 1, 3, 2, 4).reshape(3, -1, self.block_size**2)

    def summary(self) -> dict[str, float | int]:
        """Return cosine-similarity statistics for aligned Q/K/V block positions."""
        blocks = self._segment_blocks()
        normalized = torch.nn.functional.normalize(blocks, dim=2)
        labels = ("q", "k", "v")
        result: dict[str, float | int] = {
            "qkv_block_size": self.block_size,
            "qkv_aligned_block_count": blocks.shape[1],
        }

        for first, second in combinations(range(3), 2):
            similarity = (normalized[first] * normalized[second]).sum(dim=1)
            prefix = f"{labels[first]}_{labels[second]}_aligned"
            result[f"{prefix}_mean"] = similarity.mean().item()
            result[f"{prefix}_median"] = similarity.median().item()
            result[f"{prefix}_max"] = similarity.max().item()
            result[f"{prefix}_over_0_75"] = int((similarity > 0.75).sum().item())

        return result
