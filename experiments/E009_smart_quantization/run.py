"""E009 — Smart quantization methods: GPTQ, AWQ, SmoothQuant, AutoRound, NF4, GGUF, NVFP4.

Tests advanced quantization algorithms that use importance weighting, activation awareness,
or learned rounding to achieve better accuracy at low bit-widths.
"""

import json
from pathlib import Path

import torch

from genweight import ModelLoader, run_full_quantization_suite


MODEL_NAME = "gpt2"
PARAMETER_NAME = "h.0.attn.c_attn.weight"


def compute_hessian_proxy(weight: torch.Tensor) -> torch.Tensor:
    """Compute a proxy for Hessian diagonal using weight magnitude and gradient flow."""
    # Simple proxy: weights with larger magnitude and their connections
    # In practice, this would come from calibration data with actual gradients
    # Here we use a heuristic: importance ~ |weight| * (1 + |weight|)
    return weight.abs() * (1 + weight.abs())


def make_serializable(obj):
    """Recursively convert tensors to lists for JSON serialization."""
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)


def main() -> None:
    result_directory = Path("results/E009_smart_quantization")
    loader = ModelLoader(MODEL_NAME)
    loader.load()
    weight = loader.get_parameter(PARAMETER_NAME).float()

    print(f"Analyzing: {PARAMETER_NAME} {tuple(weight.shape)}")
    print(f"Weight stats: mean={weight.mean():.6f}, std={weight.std():.6f}")

    # Compute importance metrics for smart quantization
    hessian_diag = compute_hessian_proxy(weight)
    hessian_diag = hessian_diag.mean(dim=0)  # Average over output features

    # Proxy activation scales (in practice from calibration data)
    activation_scales = weight.abs().mean(dim=0) * 0.1 + 0.01

    print("\nRunning full quantization suite...")
    results = run_full_quantization_suite(
        weight,
        hessian_diag=hessian_diag,
        activation_scales=activation_scales,
        group_size=128,
        block_size=64,
        iterations=100,
    )

    print("\n" + "=" * 80)
    print("QUANTIZATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Scheme':<30} {'Error %':>8} {'SNR (dB)':>10} {'Compress':>8}")
    print("-" * 80)

    for r in results:
        scheme = r.get("scheme", "unknown")
        err = r.get("relative_frobenius_error", float('inf')) * 100
        snr = r.get("snr_db", -float('inf'))
        comp = r.get("compression_ratio", 0)

        if err == float('inf'):
            print(f"{scheme:<30} {'ERROR':>8} {'ERROR':>10} {'ERROR':>8}")
        else:
            print(f"{scheme:<30} {err:>8.2f} {snr:>10.2f} {comp:>7.2f}x")

    # Find best scheme
    valid = [r for r in results if r.get("relative_frobenius_error", float('inf')) != float('inf')]
    if valid:
        best = min(valid, key=lambda x: x["relative_frobenius_error"])
        print(f"\nBest scheme: {best['scheme']} with {best['relative_frobenius_error']*100:.2f}% error")

    # Save results (make tensors serializable)
    serializable_results = make_serializable(results)

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
            "results": serializable_results,
        }, f, indent=2)

    print(f"\nResults saved to {result_directory / 'summary.json'}")


if __name__ == "__main__":
    main()