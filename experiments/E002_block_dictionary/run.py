import json
from pathlib import Path

from genweight import BlockDictionaryAnalyzer, ModelLoader


MODEL_NAME = "gpt2"
PARAMETER_NAME = "h.0.attn.c_attn.weight"
BLOCK_SIZE = 64
CODEBOOK_SIZES = (4, 8, 16, 32)


def main() -> None:
    result_directory = Path("results/E002_block_dictionary")
    loader = ModelLoader(MODEL_NAME)
    loader.load()
    matrix = loader.get_parameter(PARAMETER_NAME)
    dictionary = BlockDictionaryAnalyzer(matrix, block_size=BLOCK_SIZE)

    results = []
    print("Block Dictionary Baseline")
    for codebook_size in CODEBOOK_SIZES:
        result = dictionary.evaluate(codebook_size)
        results.append(result)
        print(
            f"codebook={codebook_size:<2} "
            f"storage={result['storage_ratio'] * 100:6.2f}% "
            f"error={result['relative_frobenius_error'] * 100:6.2f}%"
        )

    result_directory.mkdir(parents=True, exist_ok=True)
    with (result_directory / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)


if __name__ == "__main__":
    main()
