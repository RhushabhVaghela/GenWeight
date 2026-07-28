import json
from pathlib import Path

from genweight import (
    GaussianSVDBaseline,
    LowRankAnalyzer,
    ModelLoader,
    SVDAnalyzer,
    SpatialCorrelationAnalyzer,
    WeightStatistics,
    WeightVisualizer,
)


def main() -> None:
    result_directory = Path("results/E000_weight_analysis")

    loader = ModelLoader("gpt2")
    loader.load()
    weight = loader.get_parameter("h.0.attn.c_attn.weight")

    statistics_summary = WeightStatistics(weight).summary()
    svd = SVDAnalyzer(weight)
    svd_summary = svd.summary()
    gaussian_summary = GaussianSVDBaseline(weight).summary()
    correlation = SpatialCorrelationAnalyzer(weight)
    correlation_summary = correlation.summary()
    ranks = [32, 64, 128, 256, 399, 512, 662]
    low_rank = LowRankAnalyzer(weight)
    low_rank_summary = low_rank.summary_for_ranks(ranks)

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

    print("\n" + "=" * 60)
    print("Matched Gaussian Baseline")
    print("=" * 60)
    for metric in ("effective_rank", "energy_rank_90", "energy_rank_95"):
        print(f"observed_{metric:<20} : {gaussian_summary[f'observed_{metric}']}")
        print(f"gaussian_{metric:<20} : {gaussian_summary[f'gaussian_{metric}_mean']}")

    print("\n" + "=" * 60)
    print("Spatial Correlation")
    print("=" * 60)
    for key, value in correlation_summary.items():
        print(f"{key:<32} : {value}")

    print("\n" + "=" * 60)
    print("Low-Rank Reconstruction Baseline")
    print("=" * 60)
    for item in low_rank_summary:
        print(
            f"rank={item['rank']:<3} "
            f"parameters={item['parameter_ratio'] * 100:6.2f}% "
            f"error={item['relative_frobenius_error'] * 100:6.2f}%"
        )

    visualizer = WeightVisualizer(weight)
    visualizer.histogram(
        bins=100,
        save_path=result_directory / "weight_distribution.png",
    )
    visualizer.heatmap(save_path=result_directory / "weight_heatmap.png")
    svd.plot_singular_values(
        save_path=result_directory / "singular_value_spectrum.png"
    )
    correlation.plot_lag_correlations(
        save_path=result_directory / "spatial_correlation_by_lag.png"
    )
    low_rank.plot_tradeoff(
        ranks,
        save_path=result_directory / "low_rank_tradeoff.png",
    )

    result_directory.mkdir(parents=True, exist_ok=True)
    with (result_directory / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                **statistics_summary,
                **svd_summary,
                **gaussian_summary,
                **correlation_summary,
                "low_rank_baseline": low_rank_summary,
            },
            file,
            indent=2,
        )


if __name__ == "__main__":
    main()
