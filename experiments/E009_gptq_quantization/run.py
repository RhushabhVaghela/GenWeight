"""E009 — GPTQ-style quantization: Hessian-based importance for INT4.

Implements the GPTQ algorithm (Frantar et al., 2022) for weight-only quantization.
Key insight: Not all weights are equally important. Use diagonal Hessian
(second derivative of loss) as importance metric.
"""

import json
from pathlib import Path

import torch

from genweight import ModelLoader, run_full_quantization_suite


MODEL_NAME = "gpt2"
PARAMETER_NAME = "h.0.attn.c_attn.weight"


def compute_approx_hessian_diag(weight: torch.Tensor) -> torch.Tensor:
    """
    Approximate diagonal Hessian for weight importance.

    In GPTQ, the Hessian is computed from calibration data.
    Here we use a proxy: weight magnitude squared (Fisher information approximation).
    For linear layer W @ x, the output variance ~ W^2 * Var(x).
    """
    # Proxy: ||W||^2 per input channel (columns of W)
    # This approximates the sensitivity of output to each input dimension
    hessian_diag = (weight ** 2).mean(dim=0)  # [in_features]
    # Normalize
    hessian_diag = hessian_diag / hessian_diag.mean()
    return hessian_diag


def main() -> None:
    result_directory = Path("results/E009_gptq_quantization")
    loader = ModelLoader(MODEL_NAME)
    loader.load()
    weight = loader.get_parameter(PARAMETER_NAME)

    print(f"Analyzing: {PARAMETER_NAME} {tuple(weight.shape)}")

    # Compute approximate Hessian diagonal
    hessian_diag = compute_approx_hessian_diag(weight)
    print(f"Hessian diagonal stats: mean={hessian_diag.mean():.4f}, "
          f"std={hessian_diag.std():.4f}, min={hessian_diag.min():.4f}, max={hessian_diag.max():.4f}")

    # Run full quantization suite
    schemes = [
        "int4",                      # Naive INT4 (baseline)
        "int4_per_channel",          # Per-output-channel INT4
        "int4_group",                # Group INT4 (GGUF-style)
        "gptq_int4",                 # GPTQ with Hessian importance
        "awq_int4",                  # AWQ with activation importance
        "autoround_int4",            # AutoRound with rounding optimization
        "nf4",                       # NF4 (normal distribution)
        "gguf_q4_k",                 # GGUF Q4_K_M style
        "nvfp4",                     # NVFP4 microscaling
    ]

    results = run_full_quantization_suite(
        weight,
        schemes=schemes,
        hessian_diag=hessian_diag,
        activation_scales=weight.abs().mean(dim=0),  # Proxy for AWQ
        group_size=128,
        iterations=100,
    )

    print("\n" + "=" * 80)
    print("GPTQ-Style Quantization Comparison")
    print("=" * 80)
    print(f"{'Scheme':<25} {'Error(%)':>10} {'SNR(dB)':>10} {'Comp':>8} {'Notes'}")
    print("-" * 80)

    for r in results:
        scheme = r.get("scheme", "unknown")
        err = r.get("relative_frobenius_error", float('inf')) * 100
        snr = r.get("snr_db", -float('inf'))
        comp = r.get("compression_ratio", 0)
        notes = ""
        if "group_size" in r:
            notes += f" gs={r['group_size']}"
        if "dampening" in r:
            notes += f" damp={r['dampening']}"
        if "iterations" in r:
            notes += f" it={r['iterations']}"
        print(f"{scheme:<25} {err:>10.2f} {snr:>10.2f} {comp:>8.2f}x {notes}")

    # Find best scheme
    valid = [r for r in results if r.get("relative_frobenius_error", float('inf')) != float('inf')]
    if valid:
        best = min(valid, key=lambda x: x["relative_frobenius_error"])
        print(f"\nBest scheme: {best['scheme']} with {best['relative_frobenius_error']*100:.2f}% error")

    result_directory.mkdir(parents=True, exist_ok=True)
    with (result_directory / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL_NAME,
            "parameter": PARAMETER_NAME,
            "weight_shape": list(weight.shape),
            "hessian_stats": {
                "mean": hessian_diag.mean().item(),
                "std": hessian_diag.std().item(),
                "min": hessian_diag.min().item(),
                "max": hessian_diag.max().item(),
            },
            "results": results,
        }, f, indent=2)

    print(f"\nResults saved to {result_directory / 'summary.json'}")


if __name__ == "__main__":
    main()