"""E020 — Mixed-precision quantization (sensitivity-aware bit allocation).

Key insight from E017/E019: Not all layers are equally sensitive.
MLP c_proj layers have 4x higher error than attention layers.
Mixed-precision allocates more bits to sensitive layers.

Strategies tested:
  1. Uniform NVFP4 (all layers 4-bit) — baseline
  2. Uniform INT8 (all layers 8-bit) — baseline
  3. Mixed A: Sensitive layers (mlp.c_proj) at INT8, rest at NVFP4
  4. Mixed B: Sensitive layers at INT8, medium at INT4_group, rest at NVFP4
  5. Mixed C: Ranked by Frobenius error from E017 — top-K sensitive at INT8, rest at NVFP4
  6. Optimal Mixed: Binary search to find the cheapest config that keeps perplexity < 2x baseline

We evaluate each strategy with:
  - Total bits (memory footprint)
  - Average compression ratio
  - End-to-end perplexity
"""

import sys
import os
import json
import math
import torch
import torch.nn as nn
from collections import OrderedDict

sys.path = [p for p in sys.path if 'hermes' not in p.lower()]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer
from genweight.quantization import (
    quantize_int8_per_channel,
    quantize_int4_group,
    quantize_fp8,
    quantize_nvfp4,
    quantize_gguf_q4_k,
    compute_metrics,
    make_serializable,
)


# Same test corpus as E019
TEST_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence has transformed natural language processing.",
    "Markets reflect the collective expectations of millions of participants.",
    "The theory of relativity describes how gravity bends spacetime.",
    "To be or not to be that is the question whether tis nobler.",
    "Climate scientists warn about rising global temperatures.",
    "Quantum computers leverage superposition for parallel computation.",
    "The GDP grew by three percent in the last quarter of the year.",
    "Neural networks learn hierarchical features from raw data inputs.",
    "The symphony orchestra performed Beethoven's ninth to a full house.",
    "Innovation drives economic growth through creative destruction.",
    "The human brain processes visual information in the occipital lobe.",
    "Time flies like an arrow but fruit flies like a banana.",
    "Shall I compare thee to a summer's day thou art more lovely.",
    "Knowledge transfers across domains when structural similarities exist.",
    "The equilibrium price balances supply and demand in free markets.",
    "Photosynthesis converts solar energy into chemical energy in plants.",
    "Every cloud has a silver lining as the old saying wisely goes.",
    "The beauty of mathematics lies in its elegant simplicity and power.",
    "Technology amplifies human capability but requires ethical oversight.",
    "The novel explores themes of alienation identity and belonging.",
    "Programming languages provide abstractions for managing complexity.",
    "Sound waves propagate through air at approximately 343 meters per second.",
    "The French Revolution transformed European politics and society.",
    "DNA contains the genetic instructions for building living organisms.",
]


def compute_perplexity(model, tokenizer, texts):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text, return_tensors="pt")
            if ids.shape[1] < 5:
                continue
            outputs = model(ids, labels=ids)
            num_tokens = ids.shape[1]
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


def apply_mixed_precision(model, layer_schemes):
    """Apply different quantization schemes to different layers.
    
    layer_schemes: dict mapping parameter name -> scheme name
    Supported schemes: "int8_pc", "fp8", "nvfp4", "int4_g64", "int4_g32", "gguf_q4k", "skip"
    """
    scheme_fns = {
        "int8_pc": lambda w: quantize_int8_per_channel(w)[0],
        "fp8": quantize_fp8,
        "nvfp4": lambda w: quantize_nvfp4(w, group_size=16)[0],
        "int4_g64": lambda w: quantize_int4_group(w, group_size=64)[0],
        "int4_g32": lambda w: quantize_int4_group(w, group_size=32)[0],
        "gguf_q4k": lambda w: quantize_gguf_q4_k(w, block_size=256)[0],
        "skip": lambda w: w,
    }
    
    bits_map = {
        "int8_pc": 8, "fp8": 8, "nvfp4": 4, "int4_g64": 4, "int4_g32": 4,
        "gguf_q4k": 4, "skip": 32,
    }
    
    total_bits = 0
    total_params = 0
    param_errors = {}
    
    for name, param in model.named_parameters():
        if param.ndim != 2 or "weight" not in name or param.numel() < 10000:
            continue
        
        scheme = layer_schemes.get(name, "nvfp4")
        original = param.data.clone().float()
        quantized = scheme_fns[scheme](original)
        param.data = quantized.to(param.dtype)
        
        m = compute_metrics(original, quantized)
        param_errors[name] = {
            "scheme": scheme,
            "relative_frobenius_error": m.relative_frobenius_error,
            "snr_db": m.snr_db,
        }
        
        bits = bits_map[scheme]
        total_bits += param.numel() * bits
        total_params += param.numel()
    
    avg_bits = total_bits / total_params
    compression = 32.0 / avg_bits
    return compression, avg_bits, param_errors


def get_layer_sensitivity(model):
    """Measure per-layer sensitivity via NVFP4 Frobenius error."""
    sensitivities = {}
    for name, param in model.named_parameters():
        if param.ndim != 2 or "weight" not in name or param.numel() < 10000:
            continue
        weight = param.data.clone().float()
        dequant = quantize_nvfp4(weight, group_size=16)[0]
        m = compute_metrics(weight, dequant)
        sensitivities[name] = m.relative_frobenius_error
    return sensitivities


def main():
    print("=" * 80)
    print("E020: Mixed-Precision Quantization")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Baseline
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    _, base_ppl = compute_perplexity(model, tokenizer, TEST_TEXTS)
    print(f"Baseline perplexity: {base_ppl:.2f}")
    del model

    # Step 1: Measure per-layer sensitivity
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    sensitivities = get_layer_sensitivity(model)
    del model

    print("\nPer-layer sensitivity (NVFP4 error %):")
    sorted_layers = sorted(sensitivities.items(), key=lambda x: -x[1])
    for name, err in sorted_layers:
        print(f"  {name:<55} {err*100:>6.2f}%")

    # Define strategies
    layer_names = list(sensitivities.keys())
    mlp_cproj = [n for n in layer_names if "mlp.c_proj" in n]
    mlp_cfc = [n for n in layer_names if "mlp.c_fc" in n]
    attn_layers = [n for n in layer_names if "attn" in n]
    other = [n for n in layer_names if "mlp" not in n and "attn" not in n]

    strategies = OrderedDict()

    # Strategy 1: Uniform NVFP4
    strategies["uniform_nvfp4"] = {n: "nvfp4" for n in layer_names}

    # Strategy 2: Uniform INT8
    strategies["uniform_int8"] = {n: "int8_pc" for n in layer_names}

    # Strategy 3: Uniform FP8
    strategies["uniform_fp8"] = {n: "fp8" for n in layer_names}

    # Strategy 4: Mixed A — mlp.c_proj at INT8, rest NVFP4
    strategies["mixed_a_mlp_cproj_int8"] = {
        **{n: "nvfp4" for n in layer_names},
        **{n: "int8_pc" for n in mlp_cproj},
    }

    # Strategy 5: Mixed B — sensitive (top 1/3 by error) at INT8, rest at NVFP4
    top_third = [n for n, _ in sorted_layers[:len(sorted_layers) // 3]]
    strategies["mixed_b_top_third_int8"] = {
        **{n: "nvfp4" for n in layer_names},
        **{n: "int8_pc" for n in top_third},
    }

    # Strategy 6: Mixed C — top 1/3 at FP8, middle 1/3 at NVFP4, bottom 1/3 at INT4_g32
    mid_third = [n for n, _ in sorted_layers[len(sorted_layers) // 3 : 2 * len(sorted_layers) // 3]]
    bot_third = [n for n, _ in sorted_layers[2 * len(sorted_layers) // 3:]]
    strategies["mixed_c_tiered"] = {
        **{n: "fp8" for n in top_third},
        **{n: "nvfp4" for n in mid_third},
        **{n: "int4_g32" for n in bot_third},
    }

    # Strategy 7: All-attention FP8, all-MLP NVFP4
    strategies["mixed_d_attn_fp8_mlp_nvfp4"] = {
        **{n: "fp8" for n in attn_layers},
        **{n: "nvfp4" for n in mlp_cfc + mlp_cproj},
        **{n: "fp8" for n in other},
    }

    results = {}

    for strat_name, scheme_map in strategies.items():
        print(f"\n{'=' * 60}")
        print(f"Strategy: {strat_name}")
        print(f"{'=' * 60}")

        # Count bits
        scheme_counts = {}
        for n, s in scheme_map.items():
            scheme_counts[s] = scheme_counts.get(s, 0) + 1
        print(f"  Distribution: {scheme_counts}")

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        compression, avg_bits, param_errors = apply_mixed_precision(model, scheme_map)

        avg_err = sum(e["relative_frobenius_error"] for e in param_errors.values()) / len(param_errors)
        _, ppl = compute_perplexity(model, tokenizer, TEST_TEXTS)
        ppl_delta = ((ppl - base_ppl) / base_ppl) * 100

        print(f"  Avg bits per param: {avg_bits:.2f}")
        print(f"  Compression ratio: {compression:.2f}x")
        print(f"  Avg Frobenius error: {avg_err*100:.2f}%")
        print(f"  Perplexity: {ppl:.2f} (Δ={ppl_delta:+.2f}%)")

        results[strat_name] = {
            "compression_ratio": compression,
            "avg_bits": avg_bits,
            "avg_frobenius_error": avg_err * 100,
            "perplexity": ppl,
            "perplexity_delta_pct": ppl_delta,
            "scheme_counts": scheme_counts,
        }
        del model

    # Summary
    print("\n" + "=" * 80)
    print("MIXED-PRECISION SUMMARY")
    print("=" * 80)
    print(f"  {'Strategy':<30} {'Bits':>6} {'Comp':>6} {'Frb%':>7} {'PPL':>10} {'Δ PPL%':>10}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*7} {'-'*10} {'-'*10}")
    for strat, data in results.items():
        print(f"  {strat:<30} {data['avg_bits']:>6.2f} {data['compression_ratio']:>5.2f}x "
              f"{data['avg_frobenius_error']:>7.2f} {data['perplexity']:>10.2f} {data['perplexity_delta_pct']:>+10.2f}")

    # Save
    results_dir = os.path.join(PROJECT_ROOT, "results", "E020_mixed_precision")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\nResults saved to {results_dir}/summary.json")


if __name__ == "__main__":
    main()
