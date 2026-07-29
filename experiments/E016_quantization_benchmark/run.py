"""E016 — Comprehensive quantization benchmark across all GPT-2 layers and methods.

Runs all 15 quantization schemes on all 48 weight matrices and reports
the best method per layer type.
"""

import json
from pathlib import Path

import torch

from genweight import ModelLoader, run_full_quantization_suite, make_serializable


MODEL_NAME = "gpt2"
SCHEMES = [
    "int8",
    "int4",
    "int4_group",
    "fp8",
    "fp4",
    "gptq_int4",
    "awq_int4",
    "smoothquant_int8",
    "smoothquant_int4",
    "autoround_int4",
    "nf4",
    "gguf_q4_k",
    "nvfp4",
]


def main() -> None:
    result_directory = Path("results/E016_quantization_benchmark")
    loader = ModelLoader(MODEL_NAME)
    loader.load()

    all_results = []

    for name, param in loader.model.named_parameters():
        if param.ndim != 2:
            continue

        weight = param.detach().float()
        print(f"\n--- {name} {tuple(weight.shape)} ---")

        # Compute proxy metrics for smart quantization
        hessian_diag = (weight ** 2).mean(dim=0)  # Proxy: weight magnitude squared
        hessian_diag = hessian_diag / hessian_diag.mean()
        activation_scales = weight.abs().mean(dim=0) * 0.1 + 0.01

        results = run_full_quantization_suite(
            weight,
            schemes=SCHEMES,
            hessian_diag=hessian_diag,
            activation_scales=activation_scales,
            group_size=128,
            iterations=50,
        )

        print(f"  {'Scheme':<25} {'Error(%)':>10} {'SNR(dB)':>10} {'Comp':>8}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*8}")

        layer_result = {"parameter_name": name, "shape": list(weight.shape), "schemes": []}

        for r in results:
            scheme = r.get("scheme", "unknown")
            err = r.get("relative_frobenius_error", float('inf')) * 100
            snr = r.get("snr_db", -float('inf'))
            comp = r.get("compression_ratio", 0)

            if err == float('inf'):
                print(f"  {scheme:<25} {'ERROR':>10} {'ERROR':>10} {'ERROR':>8}")
            else:
                print(f"  {scheme:<25} {err:>10.2f} {snr:>10.2f} {comp:>7.2f}x")

            layer_result["schemes"].append(make_serializable(r))

        all_results.append(layer_result)

    # Aggregate analysis
    print("\n" + "=" * 80)
    print("BEST SCHEME PER LAYER TYPE")
    print("=" * 80)

    layer_types = {
        "wte": [],
        "wpe": [],
        "attn.c_attn": [],
        "attn.c_proj": [],
        "mlp.c_fc": [],
        "mlp.c_proj": [],
    }

    for layer in all_results:
        name = layer["parameter_name"]
        valid = [(r.get("scheme", ""), r.get("relative_frobenius_error", float('inf')))
                 for r in layer["schemes"]
                 if r.get("relative_frobenius_error", float('inf')) != float('inf')]
        if valid:
            best_scheme, best_err = min(valid, key=lambda x: x[1])
            for lt in layer_types:
                if lt in name:
                    layer_types[lt].append((name, best_scheme, best_err * 100))
                    break

    for lt, entries in layer_types.items():
        if entries:
            print(f"\n{lt}:")
            for name, scheme, err in entries:
                print(f"  {name}: {scheme} ({err:.2f}%)")

    # Overall best schemes
    print("\n" + "=" * 80)
    print("OVERALL BEST SCHEMES (by average error)")
    print("=" * 80)

    scheme_stats = {}
    for layer in all_results:
        for r in layer["schemes"]:
            scheme = r.get("scheme", "")
            err = r.get("relative_frobenius_error", float('inf'))
            if err != float('inf'):
                if scheme not in scheme_stats:
                    scheme_stats[scheme] = {"errors": [], "snrs": [], "comps": []}
                scheme_stats[scheme]["errors"].append(err * 100)
                scheme_stats[scheme]["snrs"].append(r.get("snr_db", 0))
                scheme_stats[scheme]["comps"].append(r.get("compression_ratio", 0))

    print(f"{'Scheme':<30} {'Avg Error(%)':>14} {'Min Error(%)':>14} {'Max Error(%)':>14} {'Avg SNR(dB)':>12} {'Avg Comp':>10}")
    print("-" * 100)
    for scheme, stats in sorted(scheme_stats.items()):
        if stats["errors"]:
            avg_err = sum(stats["errors"]) / len(stats["errors"])
            min_err = min(stats["errors"])
            max_err = max(stats["errors"])
            avg_snr = sum(stats["snrs"]) / len(stats["snrs"])
            avg_comp = sum(stats["comps"]) / len(stats["comps"])
            print(f"{scheme:<30} {avg_err:>14.2f} {min_err:>14.2f} {max_err:>14.2f} {avg_snr:>12.2f} {avg_comp:>10.2f}x")

    # Save results
    result_directory.mkdir(parents=True, exist_ok=True)
    with (result_directory / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(make_serializable({
            "model": MODEL_NAME,
            "schemes": SCHEMES,
            "layers": all_results,
            "layer_type_analysis": make_serializable(layer_types),
            "scheme_stats": make_serializable(scheme_stats),
        }), f, indent=2)

    print(f"\nResults saved to {result_directory / 'summary.json'}")


if __name__ == "__main__":
    main()