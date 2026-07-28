from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch


class SVDAnalyzer:
    """Analyze the singular-value spectrum of a two-dimensional weight matrix."""

    def __init__(self, tensor: torch.Tensor):
        if tensor.ndim != 2:
            raise ValueError(
                "SVDAnalyzer requires a two-dimensional weight matrix; "
                f"received a {tensor.ndim}D tensor."
            )

        self.tensor = tensor.detach().float().cpu()
        self._singular_values: torch.Tensor | None = None

    @property
    def singular_values(self) -> torch.Tensor:
        """Return singular values, computing them only once."""
        if self._singular_values is None:
            self._singular_values = torch.linalg.svdvals(self.tensor)
        return self._singular_values

    def summary(self, tolerance: float | None = None) -> dict[str, Any]:
        """Return rank and spectral-energy measurements for the matrix."""
        singular_values = self.singular_values
        max_singular_value = singular_values[0]

        if tolerance is None:
            tolerance = (
                max(self.tensor.shape)
                * torch.finfo(self.tensor.dtype).eps
                * max_singular_value.item()
            )

        numerical_rank = int((singular_values > tolerance).sum().item())
        energy = singular_values.square()
        energy_share = energy / energy.sum()
        spectral_entropy = -(energy_share * energy_share.clamp_min(1e-12).log()).sum()
        effective_rank = torch.exp(spectral_entropy).item()
        cumulative_energy = torch.cumsum(energy_share, dim=0)

        return {
            "numerical_rank": numerical_rank,
            "effective_rank": effective_rank,
            "largest_singular_value": max_singular_value.item(),
            "smallest_singular_value": singular_values[-1].item(),
            "condition_number": (max_singular_value / singular_values[-1]).item(),
            "energy_rank_90": self._rank_at_energy(cumulative_energy, 0.90),
            "energy_rank_95": self._rank_at_energy(cumulative_energy, 0.95),
            "energy_rank_99": self._rank_at_energy(cumulative_energy, 0.99),
        }

    @staticmethod
    def _rank_at_energy(cumulative_energy: torch.Tensor, threshold: float) -> int:
        """Return the smallest rank retaining at least ``threshold`` energy."""
        return int(torch.searchsorted(cumulative_energy, threshold).item() + 1)

    def plot_singular_values(
        self,
        save_path: str | Path | None = None,
        show: bool = False,
    ) -> None:
        """Plot the singular-value spectrum on a logarithmic y-axis."""
        values = self.singular_values.numpy()

        figure, axis = plt.subplots(figsize=(10, 6))
        axis.plot(range(1, len(values) + 1), values)
        axis.set_yscale("log")
        axis.set_title("Singular Value Spectrum")
        axis.set_xlabel("Singular value index")
        axis.set_ylabel("Singular value (log scale)")
        axis.grid(alpha=0.3)
        figure.tight_layout()

        if save_path is not None:
            output_path = Path(save_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(output_path, dpi=300)

        if show:
            plt.show()

        plt.close(figure)
