from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from .svd import SVDAnalyzer


class LowRankAnalyzer:
    """Estimate low-rank reconstruction error and storage cost from an SVD."""

    def __init__(self, tensor: torch.Tensor):
        if tensor.ndim != 2:
            raise ValueError("LowRankAnalyzer requires a two-dimensional weight matrix.")

        self.tensor = tensor.detach().float().cpu()
        self._singular_values = SVDAnalyzer(self.tensor).singular_values

    def summary_for_ranks(self, ranks: list[int]) -> list[dict[str, float | int]]:
        """Return reconstruction and parameter-cost estimates for each rank."""
        maximum_rank = min(self.tensor.shape)
        total_energy = self._singular_values.square().sum()
        cumulative_energy = torch.cumsum(self._singular_values.square(), dim=0)
        rows, columns = self.tensor.shape
        original_parameters = rows * columns
        results = []

        for rank in sorted(set(ranks)):
            if not 1 <= rank <= maximum_rank:
                raise ValueError(f"rank must be between 1 and {maximum_rank}; received {rank}.")

            retained_energy = cumulative_energy[rank - 1]
            relative_frobenius_error = torch.sqrt(
                (total_energy - retained_energy).clamp_min(0) / total_energy
            ).item()
            factor_parameters = rank * (rows + columns + 1)

            results.append(
                {
                    "rank": rank,
                    "retained_energy": (retained_energy / total_energy).item(),
                    "relative_frobenius_error": relative_frobenius_error,
                    "factor_parameters": factor_parameters,
                    "parameter_ratio": factor_parameters / original_parameters,
                    "compression_ratio": original_parameters / factor_parameters,
                }
            )

        return results

    def plot_tradeoff(
        self,
        ranks: list[int],
        save_path: str | Path | None = None,
    ) -> None:
        """Save reconstruction error versus low-rank storage cost."""
        results = self.summary_for_ranks(ranks)
        parameter_percentages = [100 * item["parameter_ratio"] for item in results]
        errors = [100 * item["relative_frobenius_error"] for item in results]

        figure, axis = plt.subplots(figsize=(10, 6))
        axis.plot(parameter_percentages, errors, marker="o")
        for item, x_value, y_value in zip(results, parameter_percentages, errors):
            axis.annotate(str(item["rank"]), (x_value, y_value), xytext=(4, 4), textcoords="offset points")
        axis.set_title("Low-Rank Storage vs. Reconstruction Error")
        axis.set_xlabel("Stored parameters (% of dense matrix)")
        axis.set_ylabel("Relative Frobenius error (%)")
        axis.grid(alpha=0.3)
        figure.tight_layout()

        if save_path is not None:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_path, dpi=300)

        plt.close(figure)
