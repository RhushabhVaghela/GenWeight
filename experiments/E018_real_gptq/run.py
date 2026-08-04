"""E018 — Real GPTQ: Compute Hessian H=X^T X from calibration activations.

Previous experiments (E009, E016) used a weight-magnitude proxy for the
Hessian diagonal, which is why GPTQ underperformed (24-30% error).
The real GPTQ uses the true Hessian computed from calibration data:
    H = sum over calibration sequences: X^T X
where X is the input to each linear layer (rows = tokens, cols = features).

Experiment plan:
  1. Collect input activations for each weight matrix using forward hooks.
  2. Compute Hessian as H = X^T X over all calibration tokens.
  3. Use H as importance metric for sample-wise INT4 quantization.
  4. Compare true-Hessian GPTQ vs proxy-Hessian GPTQ vs NVFP4 baseline.

Key insight: the true Hessian diagonal tells us WHICH INPUT FEATURES
are important, not which weights are large. This should dramatically
improve GPTQ's performance on GPT-2.
"""

import sys
import os
import json
import torch
import torch.nn as nn

sys.path = [p for p in sys.path if 'hermes' not in p.lower()]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer
from genweight.quantization import (
    quantize_int8_per_channel,
    quantize_int4_per_channel,
    quantize_int4_group,
    quantize_nvfp4,
    compute_metrics,
    make_serializable,
    _get_weight_format,
    _restore_weight_format,
)


# Calibration sentences
CALIB_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "In a world of artificial intelligence, language models learn to predict.",
    "The stock market appears to have a predictable structure at first glance.",
    "Natural language processing has improved significantly in recent years.",
    "The theory of general relativity describes gravity as curvature of spacetime.",
    "Climate change poses significant challenges for future generations.",
    "The development of quantum computing could revolutionize technology.",
    "To be or not to be, that is the question whether tis nobler in the mind.",
    "The principles of supply and demand are fundamental to market economics.",
    "Deep neural networks learn hierarchical representations of data.",
    "Music is the universal language of mankind, expressing emotions beyond words.",
    "The sun rises in the east and sets in the west, a daily celestial dance.",
    "Innovation drives progress, pushing boundaries of what is possible.",
    "The human brain contains approximately 86 billion neurons.",
    "Time flies like an arrow, but fruit flies like a banana.",
    "The journey of a thousand miles begins with a single step.",
    "Knowledge is power, and sharing it makes the world a better place.",
    "The beauty of mathematics lies in its ability to describe nature.",
    "Technology should serve humanity, not the other way around.",
    "Every cloud has a silver lining, as the old proverb wisely says.",
]


def collect_hessians(model, tokenizer):
    """Run calibration data and compute H = X^T X for each weight matrix.
    Uses forward hooks to capture inputs to each Linear/Conv1D layer.
    """
    hook_data = {}
    handles = []

    def make_hook(layer_name):
        def hook_fn(module, input, output):
            # Input is a tuple; take first element
            x = input[0]
            if x.ndim > 2:
                x = x.reshape(-1, x.shape[-1])
            x = x.detach().float()  # [num_tokens, in_features]
            if x.shape[0] == 0:
                return
            # Accumulate H = X^T X
            hessian_chunk = x.T @ x  # [in_features, in_features]
            if layer_name not in hook_data:
                hook_data[layer_name] = {
                    "hessian": hessian_chunk,
                    "count": x.shape[0],
                }
            else:
                hook_data[layer_name]["hessian"] += hessian_chunk
                hook_data[layer_name]["count"] += x.shape[0]
        return hook_fn

    # Register hooks for each weight module
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None and module.weight.ndim == 2:
            h = module.register_forward_hook(make_hook(name))
            handles.append(h)

    # Run calibration
    model.eval()
    with torch.no_grad():
        for sent in CALIB_SENTENCES:
            ids = tokenizer.encode(sent, return_tensors="pt")
            model(ids)

    # Remove hooks
    for h in handles:
        h.remove()

    return hook_data


def quantize_gptq_true_hessian(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    group_size: int = 128,
    dampening: float = 0.01,
) -> tuple[torch.Tensor, dict]:
    """GPTQ with the actual Hessian H=X^T X from calibration data.
    
    Uses the Hessian diagonal as importance weighting for group-wise INT4.
    """
    weight, was_transposed = _get_weight_format(weight)
    out_features, in_features = weight.shape
    weight = weight.float()
    
    # Extract Hessian diagonal (importance of each input feature)
    hessian_diag = torch.diag(hessian).clamp_min(0)
    
    # Normalize
    num_groups = (in_features + group_size - 1) // group_size
    dequantized = torch.zeros_like(weight)
    
    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, in_features)
        group_weight = weight[:, start:end]
        group_hessian = hessian_diag[start:end]
        
        # Importance = 1/sqrt(H) — protect features with high Hessian
        importance = 1.0 / torch.sqrt(group_hessian + dampening)
        importance = importance / importance.mean()
        
        # Scale based on importance-weighted weights
        weighted = group_weight * importance.unsqueeze(0)
        scale = weighted.abs().max().item() / 7.0
        scale = max(scale, 1e-8)
        
        quantized = (group_weight / scale).round().clamp(-8, 7)
        dequantized[:, start:end] = quantized * scale
    
    dequantized = _restore_weight_format(dequantized, was_transposed)
    
    return dequantized, {
        "scheme": "gptq_true_hessian_int4",
        "group_size": group_size,
        "dampening": dampening,
    }


def main():
    print("=" * 80)
    print("E018: Real GPTQ with True Hessian from Calibration Data")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    print(f"Model loaded. Calibration: {len(CALIB_SENTENCES)} sentences.")

    # Step 1: Collect Hessians.
    print("\nCollecting Hessians via forward hooks...")
    hook_data = collect_hessians(model, tokenizer)
    print(f"Collected Hessians for {len(hook_data)} layers.")
    
    for name in sorted(hook_data.keys()):
        d = hook_data[name]
        diag = torch.diag(d["hessian"])
        print(f"  {name}: shape={list(d['hessian'].shape)}, tokens={d['count']}, "
              f"diag: min={diag.min():.4f}, max={diag.max():.4f}, mean={diag.mean():.4f}")

    # Step 2: Run real GPTQ vs baselines on attention layers (most interesting)
    print("\n" + "=" * 80)
    print("Comparing: GPTQ (true Hessian) vs GPTQ (proxy) vs NVFP4 vs INT8 per-channel")
    print("=" * 80)

    all_results = {}

    for name, param in model.named_parameters():
        if param.ndim != 2 or "weight" not in name or param.numel() < 10000:
            continue
        if name not in hook_data:
            continue

        weight = param.data.clone().float()
        hessian = hook_data[name]["hessian"]
        
        results = []
        
        # 1. GPTQ with true Hessian
        dequant, meta = quantize_gptq_true_hessian(weight, hessian, group_size=128)
        m = compute_metrics(weight, dequant)
        results.append({
            "scheme": "gptq_true_hessian_int4",
            "relative_frobenius_error": m.relative_frobenius_error,
            "max_absolute_error": m.max_absolute_error,
            "snr_db": m.snr_db,
            "compression_ratio": 4.0,
        })
        
        # 2. GPTQ with proxy Hessian (weight magnitude)
        proxy_diag = weight.abs().sum(dim=0) ** 2
        proxy_diag_2d = torch.diag(proxy_diag)
        dequant_proxy, meta = quantize_gptq_true_hessian(weight, proxy_diag_2d, group_size=128)
        m = compute_metrics(weight, dequant_proxy)
        results.append({
            "scheme": "gptq_proxy_int4",
            "relative_frobenius_error": m.relative_frobenius_error,
            "max_absolute_error": m.max_absolute_error,
            "snr_db": m.snr_db,
            "compression_ratio": 4.0,
        })
        
        # 3. NVFP4 (best 4-bit from E016)
        dequant, meta = quantize_nvfp4(weight, group_size=16)
        m = compute_metrics(weight, dequant)
        results.append({
            "scheme": "nvfp4",
            "relative_frobenius_error": m.relative_frobenius_error,
            "max_absolute_error": m.max_absolute_error,
            "snr_db": m.snr_db,
            "compression_ratio": 4.0,
        })
        
        # 4. INT8 per-channel (best 8-bit)
        dequant, scales = quantize_int8_per_channel(weight)
        m = compute_metrics(weight, dequant)
        results.append({
            "scheme": "int8_per_channel",
            "relative_frobenius_error": m.relative_frobenius_error,
            "max_absolute_error": m.max_absolute_error,
            "snr_db": m.snr_db,
            "compression_ratio": 4.0,
        })
        
        all_results[name] = {
            "shape": list(weight.shape),
            "schemes": results,
        }
        
        print(f"\n--- {name} {tuple(weight.shape)} ---")
        print(f"  {'Scheme':<30} {'Error(%)':>10} {'SNR(dB)':>10}")
        print(f"  {'-'*30} {'-'*10} {'-'*10}")
        for r in sorted(results, key=lambda x: x["relative_frobenius_error"]):
            err_pct = r["relative_frobenius_error"] * 100
            print(f"  {r['scheme']:<30} {err_pct:>10.2f} {r['snr_db']:>10.2f}")

    # Aggregate
    print("\n" + "=" * 80)
    print("AGGREGATE (AVERAGE ERROR % ACROSS ALL LAYERS)")
    print("=" * 80)
    all_layers = list(all_results.values())
    scheme_names = [r["scheme"] for r in all_layers[0]["schemes"]]
    for scheme in scheme_names:
        errors = []
        for layer_data in all_layers:
            for r in layer_data["schemes"]:
                if r["scheme"] == scheme:
                    errors.append(r["relative_frobenius_error"] * 100)
                    break
        avg = sum(errors) / len(errors)
        min_e = min(errors)
        max_e = max(errors)
        print(f"  {scheme:<30} avg={avg:>7.2f}%  min={min_e:>7.2f}%  max={max_e:>7.2f}%")

    # Save
    results_dir = os.path.join(PROJECT_ROOT, "results", "E018_real_gptq")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\nResults saved to {results_dir}/summary.json")


if __name__ == "__main__":
    main()
