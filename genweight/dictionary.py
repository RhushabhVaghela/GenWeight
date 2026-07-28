from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.cluster import KMeans


class BlockDictionaryAnalyzer:
    """Fit a learned codebook to fixed-size blocks of a weight matrix."""

    def __init__(self, tensor: torch.Tensor, block_size: int = 64):
        if tensor.ndim != 2:
            raise ValueError("BlockDictionaryAnalyzer requires a two-dimensional matrix.")
        if tensor.shape[0] % block_size or tensor.shape[1] % block_size:
            raise ValueError("matrix dimensions must be divisible by block_size.")

        self.tensor = tensor.detach().float().cpu()
        self.block_size = block_size

    def _blocks(self) -> np.ndarray:
        rows, columns = self.tensor.shape
        row_blocks = rows // self.block_size
        column_blocks = columns // self.block_size
        blocks = self.tensor.reshape(
            row_blocks,
            self.block_size,
            column_blocks,
            self.block_size,
        )
        return blocks.permute(0, 2, 1, 3).reshape(-1, self.block_size**2).numpy()

    def evaluate(
        self, codebook_size: int, random_state: int = 0, n_init: int = 5
    ) -> dict[str, float | int]:
        """Fit a codebook and report its reconstruction and storage cost."""
        blocks = self._blocks()
        if not 1 < codebook_size <= len(blocks):
            raise ValueError(f"codebook_size must be between 2 and {len(blocks)}.")

        clustering = KMeans(
            n_clusters=codebook_size,
            random_state=random_state,
            n_init=n_init,
        ).fit(blocks)
        reconstruction = clustering.cluster_centers_[clustering.labels_]
        relative_error = np.linalg.norm(blocks - reconstruction) / np.linalg.norm(blocks)
        index_bits = math.ceil(math.log2(codebook_size))
        dense_bits = self.tensor.numel() * 32
        dictionary_bits = codebook_size * self.block_size**2 * 32
        assignment_bits = len(blocks) * index_bits

        return {
            "codebook_size": codebook_size,
            "block_size": self.block_size,
            "block_count": len(blocks),
            "relative_frobenius_error": float(relative_error),
            "codebook_parameters": codebook_size * self.block_size**2,
            "index_bits_per_block": index_bits,
            "storage_ratio": (dictionary_bits + assignment_bits) / dense_bits,
            "compression_ratio": dense_bits / (dictionary_bits + assignment_bits),
        }
