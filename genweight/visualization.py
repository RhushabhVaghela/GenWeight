from pathlib import Path

import matplotlib.pyplot as plt
import torch


class WeightVisualizer:
    """
    Visualization utilities for weight tensors.
    """

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor.float().cpu()

    def histogram(
        self,
        bins: int = 100,
        save_path: str | None = None,
    ):
        """
        Plot histogram of all weights.
        """

        values = self.tensor.flatten().numpy()

        plt.figure(figsize=(10, 6))

        plt.hist(values, bins=bins)

        plt.title("Weight Distribution")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")

        plt.grid(alpha=0.3)

        if save_path:
            Path(save_path).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            plt.savefig(save_path, dpi=300)

        plt.show()

        plt.close()

    def heatmap(
        self,
        save_path: str | None = None,
    ):
        """
        Plot heatmap of first 2D layer.
        """

        # Display weight tensor heatmap
        tensor_np = self.tensor.numpy()
        if tensor_np.ndim > 2:
            weights = tensor_np[0]
            while weights.ndim > 2:
                weights = weights[0]
        else:
            weights = tensor_np

        plt.figure(figsize=(10, 8))

        plt.imshow(weights, cmap="viridis", aspect="auto")

        plt.colorbar()
        plt.title("Weight Heatmap")

        if save_path:
            Path(save_path).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            plt.savefig(save_path, dpi=300)

        plt.show()

        plt.close()