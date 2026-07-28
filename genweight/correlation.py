from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


class SpatialCorrelationAnalyzer:
    """Measure local correlations in a two-dimensional weight matrix."""

    def __init__(self, tensor: torch.Tensor):
        if tensor.ndim != 2:
            raise ValueError(
                "SpatialCorrelationAnalyzer requires a two-dimensional weight matrix."
            )
        self.tensor = tensor.detach().float().cpu()

    @staticmethod
    def _pearson_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
        """Return Pearson correlation for two equally shaped tensors."""
        first = first.flatten()
        second = second.flatten()
        first = first - first.mean()
        second = second - second.mean()
        denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)

        if denominator == 0:
            return 0.0

        return (torch.dot(first, second) / denominator).item()

    def summary(self) -> dict[str, float]:
        """Return correlations between immediate matrix neighbors."""
        return {
            "horizontal_neighbor_correlation": self._pearson_correlation(
                self.tensor[:, :-1], self.tensor[:, 1:]
            ),
            "vertical_neighbor_correlation": self._pearson_correlation(
                self.tensor[:-1, :], self.tensor[1:, :]
            ),
            "diagonal_neighbor_correlation": self._pearson_correlation(
                self.tensor[:-1, :-1], self.tensor[1:, 1:]
            ),
        }

    def correlation_by_lag(self, max_lag: int = 32) -> dict[str, list[float]]:
        """Measure horizontal and vertical correlation across increasing lags."""
        if max_lag < 1:
            raise ValueError("max_lag must be at least 1.")

        maximum = min(max_lag, self.tensor.shape[0] - 1, self.tensor.shape[1] - 1)
        horizontal = []
        vertical = []

        for lag in range(1, maximum + 1):
            horizontal.append(
                self._pearson_correlation(self.tensor[:, :-lag], self.tensor[:, lag:])
            )
            vertical.append(
                self._pearson_correlation(self.tensor[:-lag, :], self.tensor[lag:, :])
            )

        return {"horizontal": horizontal, "vertical": vertical}

    def plot_lag_correlations(
        self,
        max_lag: int = 32,
        save_path: str | Path | None = None,
    ) -> None:
        """Save a plot of horizontal and vertical correlation by lag."""
        correlations = self.correlation_by_lag(max_lag)
        lags = range(1, len(correlations["horizontal"]) + 1)

        figure, axis = plt.subplots(figsize=(10, 6))
        axis.plot(lags, correlations["horizontal"], label="Horizontal")
        axis.plot(lags, correlations["vertical"], label="Vertical")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title("Weight Correlation by Matrix Distance")
        axis.set_xlabel("Lag")
        axis.set_ylabel("Pearson correlation")
        axis.legend()
        axis.grid(alpha=0.3)
        figure.tight_layout()

        if save_path is not None:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_path, dpi=300)

        plt.close(figure)
