import json
from pathlib import Path

from genweight import ModelLoader, QKVSimilarityAnalyzer


MODEL_NAME = "gpt2"
PARAMETER_NAME = "h.0.attn.c_attn.weight"
RESIDUAL_RANKS = [0, 4, 8, 16, 32, 48, 64]


def main() -> None:
    result_directory = Path("results/E005_qk_residual_compression")
    loader = ModelLoader(MODEL_NAME)
    loader.load()
    matrix = loader.get_parameter(PARAMETER_NAME)
    report = QKVSimilarityAnalyzer(matrix).qk_residual_compression_report(
        RESIDUAL_RANKS
    )

    print(f"Selected Q/K heads: {report['selected_qk_heads']}")
    print("Q-to-K Transform + Low-Rank Residual")
    for result in report["results"]:
        print(
            f"residual_rank={result['residual_rank']:<2} "
            f"storage={result['parameter_ratio'] * 100:6.2f}% "
            f"error={result['relative_frobenius_error'] * 100:6.2f}%"
        )

    result_directory.mkdir(parents=True, exist_ok=True)
    with (result_directory / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


if __name__ == "__main__":
    main()
