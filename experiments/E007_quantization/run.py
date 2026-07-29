"""E007 — Quantization sensitivity analysis across all GPT-2 layers.

Measures how much reconstruction error each quantization scheme introduces,
and whether some layers are more robust than others.
"""

import json
from pathlib import Path

import torch

from genweight import ModelLoader, run_quantization_suite


MODEL_NAME = "gpt2"
QUANTIZATION_SCHEMES = [
    "int8",
    "int8_per_channel",
    "int4",
    "int4_per_channel",
    "int4_group",
    "fp8",
    "fp4",
]


def main() -> None:
    result_directory = Path("results/E007_quantization")
    loader = ModelLoader(MODEL_NAME)
    loader.load()

    all_results = []

    for name, param in loader.model.named_parameters():
        if param.ndim != 2:
            continue

        weight = param.detach().cpu().float()
        print(f"\n{'=' * 60}")
        print(f"Analyzing: {name} {tuple(weight.shape)}")
        print(f"{'=' * 60}")

        results = run_quantization_suite(weight, QUANTIZATION_SCHEMES, group_size=64)

        layer_results = {
            "parameter_name": name,
            "shape": list(weight.shape),
            "parameters": weight.numel(),
            "schemes": [],
        }

        for r in results:
            scheme_result = {
                "scheme": r.scheme,
                "relative_frobenius_error": r.relative_frobenius_error,
                "max_absolute_error": r.max_absolute_error,
                "snr_db": r.snr_db,
                "compression_ratio": r.compression_ratio,
            }
            layer_results["schemes"].append(scheme_result)

            if r.relative_frobenius_error != float('inf'):
                print(
                    f"  {r.scheme:25s}  err={r.relative_frobenius_error*100:6.2f}%  "
                    f"snr={r.snr_db:6.2f}dB  comp={r.compression_ratio:.2f}x"
                )
            else:
                print(f"  {r.scheme:25s}  ERROR")

        all_results.append(layer_results)

    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY: Average relative Frobenius error (%) per scheme across all 2D weight matrices")
    print("=" * 90)

    scheme_names = QUANTIZATION_SCHEMES
    for scheme in scheme_names:
        errors = []
        for layer in all_results:
            for s in layer["schemes"]:
                if s["scheme"].startswith(scheme) and s["relative_frobenius_error"] != float('inf'):
                    errors.append(s["relative_frobenius_error"] * 100)
        if errors:
            avg = sum(errors) / len(errors)
            mn = min(errors)
            mx = max(errors)
            print(f"  {scheme:25s}  avg={avg:6.2f}%  min={mn:6.2f}%  max={mx:6.2f}%")

    result_directory.mkdir(parents=True, exist_ok=True)
    with (result_directory / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {result_directory / 'summary.json'}")


if __name__ == "__main__":
    main()