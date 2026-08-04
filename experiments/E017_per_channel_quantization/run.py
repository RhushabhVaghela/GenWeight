"""E017 — Per-channel INT8/INT4 quantization across all GPT-2 layers.

Focus:
  - int8_per_channel: one scale per output neuron (production standard)
  - int4_per_channel: per-channel 4-bit (aggressive)
  - Compare against group-wise schemes (int4_group, gguf_q4_k, nvfp4)

Per-channel is how llama.cpp / vLLM / TensorRT-LLM actually quantize weights.
Group-wise schemes (with smaller group_size) add more scale overhead but
generally achieve lower error because the scale adapts to local weight
distributions rather than being locked to a single output channel.
"""

import sys
import os
import json
import torch

# Filter hermes paths
sys.path = [p for p in sys.path if 'hermes' not in p.lower()]

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModelForCausalLM
from genweight.quantization import (
    quantize_int8_per_channel,
    quantize_int4_per_channel,
    quantize_int4_group,
    quantize_gguf_q4_k,
    quantize_nvfp4,
    compute_metrics,
    make_serializable,
)


def run_on_weight(name: str, weight: torch.Tensor) -> list[dict]:
    """Run per-channel and group-wise schemes on one weight matrix."""
    results = []

    # 1. int8_per_channel
    dequant, scales = quantize_int8_per_channel(weight)
    m = compute_metrics(weight, dequant)
    results.append({
        "scheme": "int8_per_channel",
        "relative_frobenius_error": m.relative_frobenius_error,
        "max_absolute_error": m.max_absolute_error,
        "snr_db": m.snr_db,
        "compression_ratio": 4.0,
        "num_scales": len(scales),
    })

    # 2. int4_per_channel
    dequant, scales = quantize_int4_per_channel(weight)
    m = compute_metrics(weight, dequant)
    results.append({
        "scheme": "int4_per_channel",
        "relative_frobenius_error": m.relative_frobenius_error,
        "max_absolute_error": m.max_absolute_error,
        "snr_db": m.snr_db,
        "compression_ratio": 4.0,
        "num_scales": len(scales),
    })

    # 3. int4_group (group_size=64) — baseline group-wise
    for gs in [32, 64, 128]:
        dequant, scales = quantize_int4_group(weight, group_size=gs)
        m = compute_metrics(weight, dequant)
        results.append({
            "scheme": f"int4_group(g={gs})",
            "relative_frobenius_error": m.relative_frobenius_error,
            "max_absolute_error": m.max_absolute_error,
            "snr_db": m.snr_db,
            "compression_ratio": 4.0,
            "num_scales": len(scales),
        })

    # 4. GGUF Q4_K (block_size=256)
    dequant, meta = quantize_gguf_q4_k(weight, block_size=256)
    m = compute_metrics(weight, dequant)
    results.append({
        "scheme": "gguf_q4_k",
        "relative_frobenius_error": m.relative_frobenius_error,
        "max_absolute_error": m.max_absolute_error,
        "snr_db": m.snr_db,
        "compression_ratio": 4.0,
    })

    # 5. NVFP4 (group_size=16)  — best fine-grained from E016
    dequant, meta = quantize_nvfp4(weight, group_size=16)
    m = compute_metrics(weight, dequant)
    results.append({
        "scheme": "nvfp4",
        "relative_frobenius_error": m.relative_frobenius_error,
        "max_absolute_error": m.max_absolute_error,
        "snr_db": m.snr_db,
        "compression_ratio": 4.0,
    })

    return results


def main():
    print("=" * 80)
    print("E017: Per-Channel vs Group-Wise Quantization")
    print("=" * 80)

    print("\nLoading model: gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    print("Model loaded.\n")

    # Collect all weight matrices
    weight_matrices = []
    for name, param in model.named_parameters():
        if param.ndim == 2 and "weight" in name and param.numel() > 10000:
            weight_matrices.append((name, param.data.clone().float()))

    print(f"Found {len(weight_matrices)} weight matrices\n")

    all_results = {}
    summary_table = []

    for name, weight in weight_matrices:
        results = run_on_weight(name, weight)
        all_results[name] = {
            "shape": list(weight.shape),
            "schemes": results,
        }

        # Extract key results for table
        best_scheme = min(results, key=lambda r: r["relative_frobenius_error"])
        summary_table.append({
            "layer": name,
            "shape": list(weight.shape),
            "best_scheme": best_scheme["scheme"],
            "best_error": best_scheme["relative_frobenius_error"],
            "all_errors": {r["scheme"]: r["relative_frobenius_error"] for r in results},
        })

        # Print per-layer table
        print(f"--- {name} {tuple(weight.shape)} ---")
        print(f"  {'Scheme':<25} {'Error(%)':>10} {'SNR(dB)':>10}")
        print(f"  {'-'*25} {'-'*10} {'-'*10}")
        for r in sorted(results, key=lambda x: x["relative_frobenius_error"]):
            err_pct = r["relative_frobenius_error"] * 100
            print(f"  {r['scheme']:<25} {err_pct:>10.2f} {r['snr_db']:>10.2f}")
        print()

    # Compute aggregate stats
    print("=" * 80)
    print("AGGREGATE STATISTICS (AVERAGE ERROR % ACROSS ALL LAYERS)")
    print("=" * 80)
    scheme_names = list(summary_table[0]["all_errors"].keys())
    for scheme in scheme_names:
        errors = [s["all_errors"][scheme] * 100 for s in summary_table]
        avg = sum(errors) / len(errors)
        min_e = min(errors)
        max_e = max(errors)
        print(f"  {scheme:<25} avg={avg:>7.2f}%  min={min_e:>7.2f}%  max={max_e:>7.2f}%")

    print()
    print("=" * 80)
    print("BEST SCHEME PER LAYER")
    print("=" * 80)
    for s in summary_table:
        print(f"  {s['layer']:<45} {s['best_scheme']:<25} {s['best_error']*100:.2f}%")

    # Save results
    results_dir = os.path.join(PROJECT_ROOT, "results", "E017_per_channel_quantization")
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)

    print(f"\nResults saved to {results_dir}/summary.json")


if __name__ == "__main__":
    main()
