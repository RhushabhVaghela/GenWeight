import json
from pathlib import Path

from genweight import BlockPCABaseline, ModelLoader


MODEL_NAME = "gpt2"
PARAMETER_NAME = "h.0.attn.c_attn.weight"
BLOCK_SIZE = 64
RANKS = [8, 16, 32, 64, 128]


def main() -> None:
    result_directory = Path("results/E004_block_pca")
    loader = ModelLoader(MODEL_NAME)
    loader.load()
    matrix = loader.get_parameter(PARAMETER_NAME)
    results = BlockPCABaseline(matrix, BLOCK_SIZE).evaluate(RANKS)

    print("Block PCA Baseline")
    for result in results:
        print(
            f"rank={result['rank']:<3} "
            f"storage={result['parameter_ratio'] * 100:6.2f}% "
            f"error={result['relative_frobenius_error'] * 100:6.2f}%"
        )

    result_directory.mkdir(parents=True, exist_ok=True)
    with (result_directory / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


if __name__ == "__main__":
    main()
