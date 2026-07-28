import json
from pathlib import Path

import torch
from torch import nn

from genweight import BlockWeightGenerator, ModelLoader


MODEL_NAME = "gpt2"
PARAMETER_NAME = "h.0.attn.c_attn.weight"
BLOCK_SIZE = 64
LATENT_DIM = 32
HIDDEN_DIM = 64
TRAINING_STEPS = 500
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
SEED = 0


def extract_blocks(matrix: torch.Tensor, block_size: int) -> torch.Tensor:
    """Split a matrix into flattened non-overlapping blocks."""
    rows, columns = matrix.shape
    row_blocks = rows // block_size
    column_blocks = columns // block_size
    blocks = matrix.reshape(row_blocks, block_size, column_blocks, block_size)
    return blocks.permute(0, 2, 1, 3).reshape(-1, block_size * block_size)


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_directory = Path("results/E003_block_generator")

    loader = ModelLoader(MODEL_NAME)
    loader.load()
    matrix = loader.get_parameter(PARAMETER_NAME).float().to(device)
    targets = extract_blocks(matrix, BLOCK_SIZE)

    generator = BlockWeightGenerator(
        block_count=len(targets),
        block_size=BLOCK_SIZE,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
    ).to(device)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=LEARNING_RATE)
    loss_function = nn.MSELoss()

    print(f"Training block generator on {device}.")
    for step in range(1, TRAINING_STEPS + 1):
        block_indices = torch.randint(len(targets), (BATCH_SIZE,), device=device)
        target = targets[block_indices]

        optimizer.zero_grad()
        prediction = generator(block_indices)
        loss = loss_function(prediction, target)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 100 == 0:
            print(f"step={step:4} mse={loss.item():.8f}")

    generator.eval()
    with torch.no_grad():
        all_indices = torch.arange(len(targets), device=device)
        reconstruction = generator(all_indices)
        relative_error = (
            torch.linalg.vector_norm(reconstruction - targets)
            / torch.linalg.vector_norm(targets)
        ).item()

    parameter_count = sum(parameter.numel() for parameter in generator.parameters())
    summary = {
        "model": MODEL_NAME,
        "parameter": PARAMETER_NAME,
        "device": str(device),
        "block_size": BLOCK_SIZE,
        "block_count": len(targets),
        "latent_dim": LATENT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "training_steps": TRAINING_STEPS,
        "generator_parameters": parameter_count,
        "dense_parameters": matrix.numel(),
        "parameter_ratio": parameter_count / matrix.numel(),
        "relative_frobenius_error": relative_error,
    }

    print("\nBlock Generator Summary")
    for key, value in summary.items():
        print(f"{key:<28} : {value}")

    result_directory.mkdir(parents=True, exist_ok=True)
    with (result_directory / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    torch.save(generator.state_dict(), result_directory / "generator.pt")


if __name__ == "__main__":
    main()
