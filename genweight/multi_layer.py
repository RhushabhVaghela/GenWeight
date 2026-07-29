from __future__ import annotations

from typing import Any

import torch

from .loader import ModelLoader
from .statistics import WeightStatistics
from .svd import SVDAnalyzer
from .baselines import GaussianSVDBaseline


class MultiLayerAnalyzer:
    """Run statistics and SVD analysis across all layers of a transformer model.

    This answers whether the structural findings from a single layer (layer 0)
    generalize across the entire model — or whether deeper layers differ.
    """

    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self._model = None
        self._parameters: dict[str, torch.Tensor] = {}

    def load(self) -> None:
        """Load the model and cache all parameter tensors."""
        loader = ModelLoader(self.model_name)
        loader.load()
        self._model = loader.model
        for name, param in self._model.named_parameters():
            if param.ndim == 2:
                self._parameters[name] = param.detach().cpu()

    @property
    def parameter_names(self) -> list[str]:
        """Return the names of all 2-D weight matrices in the model."""
        return list(self._parameters.keys())

    def analyze_layer(self, parameter_name: str) -> dict[str, Any]:
        """Run statistics + SVD + Gaussian baseline on a single parameter."""
        if parameter_name not in self._parameters:
            raise KeyError(f"Parameter '{parameter_name}' not found.")

        tensor = self._parameters[parameter_name]
        stats = WeightStatistics(tensor).summary()
        svd = SVDAnalyzer(tensor)
        svd_summary = svd.summary()
        gaussian = GaussianSVDBaseline(tensor, samples=3).summary()

        return {
            "parameter_name": parameter_name,
            **stats,
            **svd_summary,
            "gaussian_effective_rank_mean": gaussian["gaussian_effective_rank_mean"],
            "effective_rank_difference": gaussian["effective_rank_difference"],
            "gaussian_condition_number_mean": gaussian["gaussian_condition_number_mean"],
            "condition_number_difference": gaussian["condition_number_difference"],
        }

    def analyze_attention_layers(self) -> list[dict[str, Any]]:
        """Analyze every c_attn weight matrix across all transformer layers."""
        attn_names = sorted(
            name for name in self._parameters
            if "attn.c_attn" in name and name.endswith(".weight")
        )
        return [self.analyze_layer(name) for name in attn_names]

    def analyze_all_parameters(self) -> list[dict[str, Any]]:
        """Analyze every 2-D weight matrix in the model."""
        return [self.analyze_layer(name) for name in self.parameter_names]
