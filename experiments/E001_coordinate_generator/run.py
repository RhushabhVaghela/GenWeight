import json
from pathlib import Path

import torch
from torch import nn

from genweight import CoordinateWeightGenerator, ModelLoader


MODEL_NAME = "gpt2"
PARAMETER_NAME = "h.0.attn.c_attn.weight"
EMBEDDING_DIM = 32
HIDDEN_DIM = 64
TRAINING_STEPS = 1_000
BATCH_SIZE = 8_192
LEARNING_RATE = 1e-3
SEED = 0


def relative_frobenius_error(
    generator: CoordinateWeightGenerator, matrix: torch.Tensor, rows: int = 16
) -> float:
    """Evaluate reconstruction error without materializing all coordinate embeddings."""
    squared_error = torch.zeros((), device=matrix.device)
    squared_target = torch.zeros((), device=matrix.device)
    column_indices = torch.arange(matrix.shape[1], device=matrix.device)

    generator.eval()
    with torch.no_grad():
        for start in range(0, matrix.shape[0], rows):
            stop = min(start + rows, matrix.shape[0])
            row_indices = torch.arange(start, stop, device=matrix.device)
            row_indices = row_indices.repeat_interleave(matrix.shape[1])
            columns = column_indices.repeat(stop - start)
            prediction = generator(row_indices, columns)
            target = matrix[start:stop].reshape(-1)
            squared_error += torch.sum((prediction - target).square())
            squared_target += torch.sum(target.square())

    return torch.sqrt(squared_error / squared_target).item()


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_directory = Path("results/E001_coordinate_generator")

    loader = ModelLoader(MODEL_NAME)
    loader.load()
    matrix = loader.get_parameter(PARAMETER_NAME).float().to(device)
    row_count, column_count = matrix.shape

    generator = CoordinateWeightGenerator(
        row_count,
        column_count,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(device)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=LEARNING_RATE)
    loss_function = nn.MSELoss()

    print(f"Training coordinate generator on {device}.")
    for step in range(1, TRAINING_STEPS + 1):
        row_indices = torch.randint(row_count, (BATCH_SIZE,), device=device)
        column_indices = torch.randint(column_count, (BATCH_SIZE,), device=device)
        target = matrix[row_indices, column_indices]

        optimizer.zero_grad()
        prediction = generator(row_indices, column_indices)
        loss = loss_function(prediction, target)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 100 == 0:
            print(f"step={step:4} mse={loss.item():.8f}")

    error = relative_frobenius_error(generator, matrix)
    parameter_count = sum(parameter.numel() for parameter in generator.parameters())
    summary = {
        "model": MODEL_NAME,
        "parameter": PARAMETER_NAME,
        "device": str(device),
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "training_steps": TRAINING_STEPS,
        "batch_size": BATCH_SIZE,
        "generator_parameters": parameter_count,
        "dense_parameters": matrix.numel(),
        "parameter_ratio": parameter_count / matrix.numel(),
        "relative_frobenius_error": error,
    }

    print("\nCoordinate Generator Summary")
    for key, value in summary.items():
        print(f"{key:<28} : {value}")

    result_directory.mkdir(parents=True, exist_ok=True)
    with (result_directory / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    torch.save(generator.state_dict(), result_directory / "generator.pt")


if __name__ == "__main__":
    main()
