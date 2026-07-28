import json
from pathlib import Path

from genweight import ModelLoader, SVDAnalyzer, WeightStatistics, WeightVisualizer


def main() -> None:
    result_directory = Path("results/E000_weight_analysis")

    loader = ModelLoader("gpt2")
    loader.load()
    weight = loader.get_parameter("h.0.attn.c_attn.weight")

    statistics_summary = WeightStatistics(weight).summary()
    svd = SVDAnalyzer(weight)
    svd_summary = svd.summary()

    print("\n" + "=" * 60)
    print("Weight Statistics")
    print("=" * 60)
    for key, value in statistics_summary.items():
        print(f"{key:<20} : {value}")

    print("\n" + "=" * 60)
    print("SVD Analysis")
    print("=" * 60)
    for key, value in svd_summary.items():
        print(f"{key:<20} : {value}")

    visualizer = WeightVisualizer(weight)
    visualizer.histogram(
        bins=100,
        save_path=result_directory / "weight_distribution.png",
    )
    visualizer.heatmap(save_path=result_directory / "weight_heatmap.png")
    svd.plot_singular_values(
        save_path=result_directory / "singular_value_spectrum.png"
    )

    result_directory.mkdir(parents=True, exist_ok=True)
    with (result_directory / "summary.json").open("w", encoding="utf-8") as file:
        json.dump({**statistics_summary, **svd_summary}, file, indent=2)


if __name__ == "__main__":
    main()
