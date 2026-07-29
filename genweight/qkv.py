from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
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

    def aligned_pairs_above(
        self,
        first_segment: int,
        second_segment: int,
        threshold: float = 0.75,
    ) -> list[dict[str, float | int]]:
        """Return aligned block locations whose cosine similarity exceeds a threshold."""
        if not 0 <= first_segment < 3 or not 0 <= second_segment < 3:
            raise ValueError("segment indices must be in the range 0 through 2.")
        if first_segment == second_segment:
            raise ValueError("the two segment indices must be different.")

        blocks = self._segment_blocks()
        normalized = torch.nn.functional.normalize(blocks, dim=2)
        similarity = (normalized[first_segment] * normalized[second_segment]).sum(dim=1)
        blocks_per_row = (self.tensor.shape[1] // 3) // self.block_size
        positions = torch.where(similarity > threshold)[0]

        return [
            {
                "row_block": position.item() // blocks_per_row,
                "column_block": position.item() % blocks_per_row,
                "cosine_similarity": similarity[position].item(),
            }
            for position in positions
        ]

    def head_similarity_matrix(
        self, first_segment: int, second_segment: int
    ) -> torch.Tensor:
        """Return cosine similarity between every output head in two Q/K/V segments."""
        if not 0 <= first_segment < 3 or not 0 <= second_segment < 3:
            raise ValueError("segment indices must be in the range 0 through 2.")

        rows, columns = self.tensor.shape
        segment_width = columns // 3
        head_count = segment_width // self.block_size
        segments = self.tensor.reshape(rows, 3, segment_width).permute(1, 0, 2)
        heads = segments.reshape(3, rows, head_count, self.block_size)
        heads = heads.permute(0, 2, 1, 3).reshape(3, head_count, -1)
        normalized = torch.nn.functional.normalize(heads, dim=2)
        return normalized[first_segment] @ normalized[second_segment].T

    def top_head_pairs(
        self, first_segment: int, second_segment: int, count: int = 10
    ) -> list[dict[str, float | int]]:
        """Return the strongest cross-segment head similarities."""
        if count < 1:
            raise ValueError("count must be at least 1.")

        similarity = self.head_similarity_matrix(first_segment, second_segment)
        pair_count = min(count, similarity.numel())
        scores, positions = torch.topk(similarity.flatten(), k=pair_count)
        head_count = similarity.shape[1]
        return [
            {
                "first_head": position.item() // head_count,
                "second_head": position.item() % head_count,
                "cosine_similarity": score.item(),
            }
            for score, position in zip(scores, positions)
        ]

    def same_index_reuse_report(
        self, source_segment: int = 0, target_segment: int = 1
    ) -> list[dict[str, float | int]]:
        """Fit one scalar per matching head and report target reconstruction residuals.

        For each head, this tests ``target ≈ scale × source``. The relative
        residual is the fraction of target L2 norm that this simple reusable
        representation fails to explain.
        """
        if not 0 <= source_segment < 3 or not 0 <= target_segment < 3:
            raise ValueError("segment indices must be in the range 0 through 2.")
        if source_segment == target_segment:
            raise ValueError("source and target segments must be different.")

        rows, columns = self.tensor.shape
        segment_width = columns // 3
        head_count = segment_width // self.block_size
        segments = self.tensor.reshape(rows, 3, segment_width).permute(1, 0, 2)
        heads = segments.reshape(3, rows, head_count, self.block_size)
        heads = heads.permute(0, 2, 1, 3).reshape(3, head_count, -1)
        source_heads = heads[source_segment]
        target_heads = heads[target_segment]
        report = []

        for head_index in range(head_count):
            source = source_heads[head_index]
            target = target_heads[head_index]
            scale = torch.dot(source, target) / torch.dot(source, source)
            residual = target - scale * source
            relative_residual = (
                torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(target)
            ).item()
            cosine_similarity = (
                torch.dot(source, target)
                / (torch.linalg.vector_norm(source) * torch.linalg.vector_norm(target))
            ).item()
            report.append(
                {
                    "head": head_index,
                    "scale": scale.item(),
                    "cosine_similarity": cosine_similarity,
                    "relative_residual": relative_residual,
                }
            )

        return report

    def same_index_linear_reuse_report(
        self, source_segment: int = 0, target_segment: int = 1
    ) -> list[dict[str, float | int]]:
        """Fit a per-head linear map from source weights to target weights.

        Each relation is ``target ≈ source × A``. ``A`` has ``head_dim²``
        parameters, compared with ``rows × head_dim`` parameters for a dense
        target head.
        """
        if not 0 <= source_segment < 3 or not 0 <= target_segment < 3:
            raise ValueError("segment indices must be in the range 0 through 2.")
        if source_segment == target_segment:
            raise ValueError("source and target segments must be different.")

        rows, columns = self.tensor.shape
        segment_width = columns // 3
        head_count = segment_width // self.block_size
        segments = self.tensor.reshape(rows, 3, segment_width).permute(1, 0, 2)
        heads = segments.reshape(3, rows, head_count, self.block_size)
        heads = heads.permute(0, 2, 1, 3)
        source_heads = heads[source_segment]
        target_heads = heads[target_segment]
        transform_parameters = self.block_size**2
        target_parameters = rows * self.block_size
        report = []

        for head_index in range(head_count):
            source = source_heads[head_index]
            target = target_heads[head_index]
            transform = torch.linalg.lstsq(source, target).solution
            residual = target - source @ transform
            relative_residual = (
                torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(target)
            ).item()
            report.append(
                {
                    "head": head_index,
                    "relative_residual": relative_residual,
                    "transform_parameters": transform_parameters,
                    "target_parameters": target_parameters,
                    "parameter_ratio": transform_parameters / target_parameters,
                }
            )

        return report

    def plot_head_similarity(
        self,
        first_segment: int,
        second_segment: int,
        save_path: str | Path | None = None,
    ) -> None:
        """Save a heatmap of cross-segment output-head similarity."""
        labels = ("Q", "K", "V")
        similarity = self.head_similarity_matrix(first_segment, second_segment)
        figure, axis = plt.subplots(figsize=(8, 6))
        image = axis.imshow(similarity.numpy(), cmap="coolwarm", vmin=-1, vmax=1)
        figure.colorbar(image, ax=axis, label="Cosine similarity")
        axis.set_title(f"{labels[first_segment]}–{labels[second_segment]} Head Similarity")
        axis.set_xlabel(f"{labels[second_segment]} head")
        axis.set_ylabel(f"{labels[first_segment]} head")
        figure.tight_layout()

        if save_path is not None:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_path, dpi=300)

        plt.close(figure)

    def qk_residual_compression_report(
        self, residual_ranks: list[int], similarity_threshold: float = 0.75
    ) -> dict[str, object]:
        """Estimate storage and error for regenerating selected K heads from Q heads.

        A selected key head is represented as ``K ≈ Q × A + U × V``. The
        transform ``A`` is always stored; ``U × V`` is a rank-limited residual.
        Query heads remain stored normally and are reused by the decoder.
        """
        rows, columns = self.tensor.shape
        segment_width = columns // 3
        head_count = segment_width // self.block_size
        segments = self.tensor.reshape(rows, 3, segment_width).permute(1, 0, 2)
        q_heads = segments[0].reshape(rows, head_count, self.block_size).permute(1, 0, 2)
        k_heads = segments[1].reshape(rows, head_count, self.block_size).permute(1, 0, 2)

        selected_heads = []
        residual_singular_values = []
        for head_index in range(head_count):
            query = q_heads[head_index]
            key = k_heads[head_index]
            cosine = (
                torch.sum(query * key)
                / (torch.linalg.vector_norm(query) * torch.linalg.vector_norm(key))
            ).item()
            if cosine >= similarity_threshold:
                transform = torch.linalg.lstsq(query, key).solution
                residual = key - query @ transform
                selected_heads.append(head_index)
                residual_singular_values.append(torch.linalg.svdvals(residual))

        if not selected_heads:
            raise ValueError("no Q/K heads met the requested similarity threshold.")

        total_squared_norm = self.tensor.square().sum()
        dense_parameters = self.tensor.numel()
        dense_key_head_parameters = rows * self.block_size
        results = []

        for rank in sorted(set(residual_ranks)):
            if not 0 <= rank <= self.block_size:
                raise ValueError(f"residual rank must be between 0 and {self.block_size}.")

            residual_squared_norm = sum(
                singular_values[rank:].square().sum()
                for singular_values in residual_singular_values
            )
            generated_parameters_per_head = self.block_size**2 + rank * (
                rows + self.block_size
            )
            stored_parameters = dense_parameters - len(selected_heads) * dense_key_head_parameters
            stored_parameters += len(selected_heads) * generated_parameters_per_head
            results.append(
                {
                    "residual_rank": rank,
                    "relative_frobenius_error": torch.sqrt(
                        residual_squared_norm / total_squared_norm
                    ).item(),
                    "stored_parameters": stored_parameters,
                    "parameter_ratio": stored_parameters / dense_parameters,
                    "compression_ratio": dense_parameters / stored_parameters,
                }
            )

        return {"selected_qk_heads": selected_heads, "results": results}
