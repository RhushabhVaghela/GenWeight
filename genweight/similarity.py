from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


class BlockSimilarityAnalyzer:
    """Measure cosine similarity among equally sized blocks of a weight matrix."""

    def __init__(self, tensor: torch.Tensor, block_size: int = 64):
        if tensor.ndim != 2:
            raise ValueError("BlockSimilarityAnalyzer requires a two-dimensional matrix.")
        if block_size < 1:
            raise ValueError("block_size must be positive.")
        if tensor.shape[0] % block_size or tensor.shape[1] % block_size:
            raise ValueError(
                f"matrix shape {tuple(tensor.shape)} is not divisible by block size {block_size}."
            )

        self.tensor = tensor.detach().float().cpu()
        self.block_size = block_size
        self._similarity_matrix: torch.Tensor | None = None

    @property
    def similarity_matrix(self) -> torch.Tensor:
        """Return the pairwise cosine-similarity matrix for all blocks."""
        if self._similarity_matrix is None:
            rows, columns = self.tensor.shape
            row_blocks = rows // self.block_size
            column_blocks = columns // self.block_size
            blocks = self.tensor.reshape(
                row_blocks,
                self.block_size,
                column_blocks,
                self.block_size,
            )
            blocks = blocks.permute(0, 2, 1, 3).reshape(-1, self.block_size**2)
            normalized_blocks = torch.nn.functional.normalize(blocks, dim=1)
            self._similarity_matrix = normalized_blocks @ normalized_blocks.T
        return self._similarity_matrix

    def summary(self) -> dict[str, float | int]:
        """Return descriptive statistics for similarity between distinct blocks."""
        similarities = self.similarity_matrix
        off_diagonal = similarities[
            torch.triu_indices(similarities.shape[0], similarities.shape[1], offset=1).unbind()
        ]

        return {
            "block_size": self.block_size,
            "block_count": similarities.shape[0],
            "mean_block_cosine_similarity": off_diagonal.mean().item(),
            "std_block_cosine_similarity": off_diagonal.std().item(),
            "max_block_cosine_similarity": off_diagonal.max().item(),
            "p95_block_cosine_similarity": torch.quantile(off_diagonal, 0.95).item(),
            "blocks_similarity_over_0_5": int((off_diagonal > 0.5).sum().item()),
            "blocks_similarity_over_0_75": int((off_diagonal > 0.75).sum().item()),
            "blocks_similarity_over_0_9": int((off_diagonal > 0.9).sum().item()),
        }

    def top_similar_pairs(self, count: int = 10) -> list[dict[str, float | int]]:
        """Return the most similar distinct block pairs and their grid coordinates."""
        if count < 1:
            raise ValueError("count must be at least 1.")

        similarities = self.similarity_matrix
        row_indices, column_indices = torch.triu_indices(
            similarities.shape[0], similarities.shape[1], offset=1
        )
        pair_scores = similarities[row_indices, column_indices]
        pair_count = min(count, pair_scores.numel())
        top_scores, top_positions = torch.topk(pair_scores, k=pair_count)
        blocks_per_row = self.tensor.shape[1] // self.block_size
        pairs = []

        for score, position in zip(top_scores, top_positions):
            first_block = row_indices[position].item()
            second_block = column_indices[position].item()
            pairs.append(
                {
                    "first_block": first_block,
                    "first_row_block": first_block // blocks_per_row,
                    "first_column_block": first_block % blocks_per_row,
                    "second_block": second_block,
                    "second_row_block": second_block // blocks_per_row,
                    "second_column_block": second_block % blocks_per_row,
                    "cosine_similarity": score.item(),
                }
            )

        return pairs

    def plot_similarity_matrix(self, save_path: str | Path | None = None) -> None:
        """Save a heatmap of pairwise block cosine similarities."""
        figure, axis = plt.subplots(figsize=(10, 8))
        image = axis.imshow(self.similarity_matrix.numpy(), cmap="coolwarm", vmin=-1, vmax=1)
        figure.colorbar(image, ax=axis, label="Cosine similarity")
        axis.set_title(f"{self.block_size}×{self.block_size} Block Similarity")
        axis.set_xlabel("Block index")
        axis.set_ylabel("Block index")
        figure.tight_layout()

        if save_path is not None:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_path, dpi=300)

        plt.close(figure)
