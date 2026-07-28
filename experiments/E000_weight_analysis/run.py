import json
from pathlib import Path

from genweight import (
    BlockSimilarityAnalyzer,
    GaussianSVDBaseline,
    LowRankAnalyzer,
    ModelLoader,
    QKVSimilarityAnalyzer,
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
    blocks = BlockSimilarityAnalyzer(weight, block_size=64)
    block_similarity_summary = blocks.summary()
    top_block_pairs = blocks.top_similar_pairs()
    qkv = QKVSimilarityAnalyzer(weight, block_size=64)
    qkv_similarity_summary = qkv.summary()
    high_qk_pairs = qkv.aligned_pairs_above(0, 1, threshold=0.75)
    top_qk_head_pairs = qkv.top_head_pairs(0, 1)

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

    print("\n" + "=" * 60)
    print("Block Similarity")
    print("=" * 60)
    for key, value in block_similarity_summary.items():
        print(f"{key:<32} : {value}")

    print("\nTop Similar Block Pairs")
    for pair in top_block_pairs:
        print(
            f"({pair['first_row_block']}, {pair['first_column_block']}) ↔ "
            f"({pair['second_row_block']}, {pair['second_column_block']}) "
            f"cosine={pair['cosine_similarity']:.4f}"
        )

    print("\n" + "=" * 60)
    print("Aligned Q/K/V Block Similarity")
    print("=" * 60)
    for key, value in qkv_similarity_summary.items():
        print(f"{key:<32} : {value}")

    print("\nHigh-Similarity Q/K Block Locations")
    for pair in high_qk_pairs:
        print(
            f"row_block={pair['row_block']:<2} "
            f"column_block={pair['column_block']:<2} "
            f"cosine={pair['cosine_similarity']:.4f}"
        )

    print("\nTop Q/K Head Pairs")
    for pair in top_qk_head_pairs:
        print(
            f"Q head={pair['first_head']:<2} "
            f"K head={pair['second_head']:<2} "
            f"cosine={pair['cosine_similarity']:.4f}"
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
    blocks.plot_similarity_matrix(
        save_path=result_directory / "block_similarity.png"
    )
    qkv.plot_head_similarity(
        0,
        1,
        save_path=result_directory / "qk_head_similarity.png",
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
                **block_similarity_summary,
                "top_block_pairs": top_block_pairs,
                **qkv_similarity_summary,
                "high_qk_pairs": high_qk_pairs,
                "top_qk_head_pairs": top_qk_head_pairs,
            },
            file,
            indent=2,
        )


if __name__ == "__main__":
    main()
