"""E006 — Multi-layer analysis across all GPT-2 attention layers.

Answers: does the structural findings from layer 0 generalize?
Specifically, is every attention layer low-rank relative to Gaussian,
and does QK-alignment persist across depth?
"""

import json
from pathlib import Path

from genweight import MultiLayerAnalyzer, QKVSimilarityAnalyzer


MODEL_NAME = "gpt2"
BLOCK_SIZE = 64


def main() -> None:
    result_directory = Path("results/E006_multi_layer")

    analyzer = MultiLayerAnalyzer(MODEL_NAME)
    analyzer.load()

    # --- Attention layer analysis ---
    attn_results = analyzer.analyze_attention_layers()

    print("=" * 70)
    print("Multi-Layer Attention (c_attn) Analysis")
    print("=" * 70)
    print(
        f"{'Layer':<30} {'Shape':<16} {'EffRank':>8} {'GaussEff':>10} "
        f"{'CondNum':>8} {'GaussCond':>10} {'Diff':>8}"
    )
    for item in attn_results:
        name = item["parameter_name"]
        shape = str(tuple(item["shape"]))
        eff = item["effective_rank"]
        gauss_eff = item["gaussian_effective_rank_mean"]
        cond = item["condition_number"]
        gauss_cond = item["gaussian_condition_number_mean"]
        diff = item["effective_rank_difference"]
        print(
            f"{name:<30} {shape:<16} {eff:>8.1f} {gauss_eff:>10.1f} "
            f"{cond:>8.1f} {gauss_cond:>10.1f} {diff:>8.1f}"
        )

    # --- QK alignment across layers ---
    print("\n" + "=" * 70)
    print("QK Same-Index Reuse Across Layers")
    print("=" * 70)
    qk_across_layers = []
    attn_param_names = [
        name for name in analyzer.parameter_names if "attn.c_attn" in name
    ]

    for name in attn_param_names:
        tensor = analyzer._parameters[name]
        qkv = QKVSimilarityAnalyzer(tensor, block_size=BLOCK_SIZE)
        reuse = qkv.same_index_reuse_report(source_segment=0, target_segment=1)

        cosines = [r["cosine_similarity"] for r in reuse]
        residuals = [r["relative_residual"] for r in reuse]
        high_sim_count = sum(1 for c in cosines if c > 0.75)

        layer_summary = {
            "parameter_name": name,
            "mean_cosine": sum(cosines) / len(cosines),
            "max_cosine": max(cosines),
            "min_cosine": min(cosines),
            "mean_residual": sum(residuals) / len(residuals),
            "min_residual": min(residuals),
            "heads_above_0.75": high_sim_count,
            "head_details": reuse,
        }
        qk_across_layers.append(layer_summary)

        print(
            f"{name:<30} "
            f"mean_cos={layer_summary['mean_cosine']:+.4f} "
            f"max_cos={layer_summary['max_cosine']:+.4f} "
            f"min_resid={layer_summary['min_residual']:.4f} "
            f"heads>0.75={high_sim_count}"
        )

    result_directory.mkdir(parents=True, exist_ok=True)
    with (result_directory / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": MODEL_NAME,
                "attention_layer_analysis": attn_results,
                "qk_reuse_across_layers": qk_across_layers,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {result_directory / 'summary.json'}")


if __name__ == "__main__":
    main()
