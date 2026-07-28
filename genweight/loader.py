from transformers import AutoModel
import torch


class ModelLoader:
    """
    Loads Hugging Face models and exposes their parameters.
    """

    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.model = None

    def load(self):
        """
        Load model into memory.
        """

        print(f"\nLoading model: {self.model_name}")

        self.model = AutoModel.from_pretrained(
            self.model_name
        )

        self.model.eval()

        print("Model loaded successfully.\n")

        return self.model

    def list_parameters(self):
        """
        Print all parameter names and shapes.
        """

        if self.model is None:
            raise RuntimeError("Load model first.")

        print("=" * 80)

        for name, param in self.model.named_parameters():

            print(
                f"{name:60}"
                f"{str(tuple(param.shape)):20}"
                f"{param.dtype}"
            )

        print("=" * 80)

    def get_parameter(self, parameter_name: str):
        """
        Return requested parameter tensor.
        """

        if self.model is None:
            raise RuntimeError("Load model first.")

        for name, param in self.model.named_parameters():

            if name == parameter_name:
                return param.detach().cpu()

        raise ValueError(f"Parameter '{parameter_name}' not found.")

    def parameter_count(self):

        if self.model is None:
            raise RuntimeError("Load model first.")

        total = sum(
            p.numel()
            for p in self.model.parameters()
        )

        return total