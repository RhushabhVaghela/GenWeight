import torch


class WeightStatistics:
    """
    Compute descriptive statistics for a weight tensor.
    """

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor.float()

    def summary(self):

        flat = self.tensor.flatten()

        stats = {
            "shape": tuple(self.tensor.shape),
            "parameters": flat.numel(),
            "mean": flat.mean().item(),
            "std": flat.std().item(),
            "variance": flat.var().item(),
            "minimum": flat.min().item(),
            "maximum": flat.max().item(),
            "median": flat.median().item(),
            "l1_norm": flat.abs().sum().item(),
            "l2_norm": torch.linalg.vector_norm(flat).item(),
            "zero_count": (flat == 0).sum().item(),
        }

        stats["sparsity"] = (
            stats["zero_count"] / stats["parameters"]
        )

        return stats