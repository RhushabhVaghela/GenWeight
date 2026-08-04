"""E019 — End-to-end perplexity evaluation.

Previous experiments measured Frobenius error on individual weight matrices.
But Frobenius error ≠ model quality. The real metric is perplexity:
how well does the quantized model predict the next token on held-out text?

Experiment:
  1. Load GPT-2, measure baseline perplexity on a text corpus.
  2. Apply INT8 per-channel quantization to ALL weight matrices.
  3. Measure quantized perplexity.
  4. Repeat for INT4 group-wise, NVFP4, and combined FP8.
  5. Compare perplexity increase to Frobenius error predictions.

Key hypothesis: Methods with low Frobenius error should have low perplexity
increase, but the relationship may not be linear — some weights (attention
output projection) matter more for perplexity than others (MLP expansion).

Test data: A diverse set of English sentences (no external dataset needed
to avoid download complexity on CPU-only environment).
"""

import sys
import os
import json
import math
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
    quantize_fp8,
    quantize_nvfp4,
    quantize_gguf_q4_k,
    make_serializable,
)


# Diverse test corpus
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
    """Compute average perplexity (token-level) over texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text, return_tensors="pt")
            # Skip very short texts
            if ids.shape[1] < 5:
                continue
            outputs = model(ids, labels=ids)
            num_tokens = ids.shape[1]
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def quantize_model_weights(model, scheme_fn):
    """Apply a quantization function to all 2D weight tensors in-place.
    Returns dict of per-layer Frobenius error.
    """
    from genweight.quantization import compute_metrics

    param_errors = {}

    for name, param in model.named_parameters():
        if param.ndim != 2 or "weight" not in name:
            continue
        if param.numel() < 10000:
            continue

        original = param.data.clone().float()
        quantized = scheme_fn(original)

        if isinstance(quantized, tuple):
            quantized = quantized[0]

        param.data = quantized.to(param.dtype)

        m = compute_metrics(original, quantized)
        param_errors[name] = {
            "relative_frobenius_error": m.relative_frobenius_error,
            "snr_db": m.snr_db,
        }

    return param_errors


def main():
    print("=" * 80)
    print("E019: End-to-End Perplexity Evaluation")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Baseline perplexity
    print("\nLoading model: gpt2 (baseline)")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    base_loss, base_ppl = compute_perplexity(model, tokenizer, TEST_TEXTS)
    print(f"Baseline: loss={base_loss:.4f}, perplexity={base_ppl:.2f}")

    results = {"baseline": {"loss": base_loss, "perplexity": base_ppl}}

    # Test each quantization scheme
    schemes = [
        ("INT8 per-channel", lambda w: quantize_int8_per_channel(w)[0]),
        ("INT4 group (g=64)", lambda w: quantize_int4_group(w, group_size=64)[0]),
        ("INT4 group (g=32)", lambda w: quantize_int4_group(w, group_size=32)[0]),
        ("FP8 (E4M3)", quantize_fp8),
        ("NVFP4", lambda w: quantize_nvfp4(w, group_size=16)[0]),
        ("GGUF Q4_K", lambda w: quantize_gguf_q4_k(w, block_size=256)[0]),
        ("INT4 per-channel", lambda w: quantize_int4_per_channel(w)[0]),
    ]

    for scheme_name, scheme_fn in schemes:
        print(f"\n{'=' * 60}")
        print(f"Testing: {scheme_name}")
        print(f"{'=' * 60}")

        # Reload model fresh each time
        model = AutoModelForCausalLM.from_pretrained("gpt2")

        # Apply quantization
        param_errors = quantize_model_weights(model, scheme_fn)

        # Compute average Frobenius error
        avg_err = sum(e["relative_frobenius_error"] for e in param_errors.values()) / len(param_errors)
        max_err = max(e["relative_frobenius_error"] for e in param_errors.values())
        print(f"  Avg Frobenius error: {avg_err*100:.2f}%, max: {max_err*100:.2f}%")

        # Measure perplexity
        loss, ppl = compute_perplexity(model, tokenizer, TEST_TEXTS)
        ppl_increase = ((ppl - base_ppl) / base_ppl) * 100
        print(f"  Loss: {loss:.4f} (baseline: {base_loss:.4f})")
        print(f"  Perplexity: {ppl:.2f} (baseline: {base_ppl:.2f})")
        print(f"  Perplexity increase: {ppl_increase:+.2f}%")

        results[scheme_name] = {
            "loss": loss,
            "perplexity": ppl,
            "perplexity_increase_pct": ppl_increase,
            "avg_frobenius_error": avg_err * 100,
            "max_frobenius_error": max_err * 100,
            "param_errors": param_errors,
        }

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary table
    print("\n" + "=" * 80)
    print("PERPLEXITY SUMMARY")
    print("=" * 80)
    print(f"  {'Scheme':<25} {'Loss':>8} {'Perplexity':>12} {'Δ PPL%':>10} {'Avg Frb%':>10}")
    print(f"  {'-'*25} {'-'*8} {'-'*12} {'-'*10} {'-'*10}")
    for scheme_name, data in results.items():
        ppl = data["perplexity"]
        if scheme_name == "baseline":
            print(f"  {scheme_name:<25} {data['loss']:>8.4f} {ppl:>12.2f} {'---':>10} {'---':>10}")
        else:
            delta = data["perplexity_increase_pct"]
            avg_e = data["avg_frobenius_error"]
            print(f"  {scheme_name:<25} {data['loss']:>8.4f} {ppl:>12.2f} {delta:>+10.2f} {avg_e:>10.2f}")

    # Save
    results_dir = os.path.join(PROJECT_ROOT, "results", "E019_perplexity")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"\nResults saved to {results_dir}/summary.json")


if __name__ == "__main__":
    main()
