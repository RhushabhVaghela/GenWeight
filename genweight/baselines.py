from __future__ import annotations

from typing import Any

import torch

from .svd import SVDAnalyzer


class GaussianSVDBaseline:
    """Compare a weight matrix's spectrum with matched Gaussian matrices.

    Each baseline has the same shape, mean, and standard deviation as the
    observed matrix. This prevents scale alone from being mistaken for learned
    structure.
    """

    def __init__(self, tensor: torch.Tensor, samples: int = 5, seed: int = 0):
        if tensor.ndim != 2:
            raise ValueError("GaussianSVDBaseline requires a two-dimensional tensor.")
        if samples < 1:
            raise ValueError("samples must be at least 1.")

        self.tensor = tensor.detach().float().cpu()
        self.samples = samples
        self.seed = seed

    def summary(self) -> dict[str, Any]:
        """Return observed SVD metrics and matched-baseline mean and deviation."""
        observed = SVDAnalyzer(self.tensor).summary()
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        baseline_summaries = []

        for _ in range(self.samples):
            gaussian = (
                torch.randn(self.tensor.shape, generator=generator)
                * self.tensor.std()
                + self.tensor.mean()
            )
            baseline_summaries.append(SVDAnalyzer(gaussian).summary())

        result: dict[str, Any] = {
            "gaussian_baseline_samples": self.samples,
            "gaussian_baseline_seed": self.seed,
        }
        for metric, observed_value in observed.items():
            values = torch.tensor(
                [item[metric] for item in baseline_summaries],
                dtype=torch.float64,
            )
            baseline_mean = values.mean().item()
            result[f"observed_{metric}"] = observed_value
            result[f"gaussian_{metric}_mean"] = baseline_mean
            result[f"gaussian_{metric}_std"] = values.std(unbiased=False).item()
            result[f"{metric}_difference"] = observed_value - baseline_mean

        return result
