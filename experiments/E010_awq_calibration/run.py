"""E010 — AWQ quantization with real calibration data.

AWQ (Activation-Aware Weight Quantization) uses activation statistics from
calibration data to protect important weights. This experiment runs actual
forward passes on a calibration dataset to collect activation scales.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from genweight import (
    ModelLoader, 
    quantize_awq_int4, 
    quantize_int4_group, 
    compute_metrics,
    quantize_matrix_smart,
)


MODEL_NAME = "gpt2"
PARAMETER_NAME = "transformer.h.0.attn.c_attn.weight"  # HF parameter name
MODULE_NAME = "transformer.h.0.attn.c_attn"
CALIBRATION_SAMPLES = 128
MAX_SEQ_LEN = 512


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


def collect_activation_scales(model, tokenizer, samples: int = 128, max_len: int = 512):
    """
    Run forward passes on calibration data to collect per-channel activation scales.

    For AWQ, we need the maximum activation magnitude per input channel
    across a representative dataset.
    """
    device = next(model.parameters()).device
    model.eval()

    # Storage for activation scales
    activation_scales = {}

    def hook_fn(name):
        def hook(module, input, output):
            # input[0] shape: [batch, seq, in_features]
            # We want per-channel max across batch and seq
            if name not in activation_scales:
                activation_scales[name] = input[0].abs().amax(dim=(0, 1)).detach().cpu()
            else:
                activation_scales[name] = torch.max(
                    activation_scales[name],
                    input[0].abs().amax(dim=(0, 1)).detach().cpu()
                )
        return hook

    # Register hooks on linear layers (including Conv1D)
    hooks = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            hooks.append(module.register_forward_hook(hook_fn(name)))

    # Prepare calibration data
    calibration_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Machine learning models require large amounts of data.",
        "Neural networks learn patterns from examples.",
        "Transformers use attention mechanisms for context.",
        "Language models predict the next token in sequence.",
        "Quantization reduces model size for deployment.",
        "Efficient inference requires hardware optimization.",
    ] * (samples // 8 + 1)

    with torch.no_grad():
        for text in calibration_texts[:samples]:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_len,
                truncation=True,
                padding=False
            ).to(device)

            model(**inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activation_scales


def main() -> None:
    result_directory = Path("results/E010_awq_calibration")

    # Load model with tokenizer for calibration
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Collect activation scales
    print(f"Collecting activation scales from {CALIBRATION_SAMPLES} samples...")
    act_scales = collect_activation_scales(model, tokenizer, CALIBRATION_SAMPLES, MAX_SEQ_LEN)

    # Get the target weight using the HF parameter name
    weight = model.get_parameter(PARAMETER_NAME).detach().float()
    print(f"Analyzing: {PARAMETER_NAME} {tuple(weight.shape)}")

    # Extract activation scales for this layer's input channels
    # c_attn weight: [768, 2304] where 2304 = 3 * 768 (QKV)
    # Input channels = 768
    if MODULE_NAME in act_scales:
        act_scale = act_scales[MODULE_NAME]
        print(f"Activation scale stats: mean={act_scale.mean():.6f}, " f"std={act_scale.std():.6f}, min={act_scale.min():.6f}, max={act_scale.max():.6f}")
    else:
        print(f"Module {MODULE_NAME} not found in activation scales, using weight proxy")
        print(f"Available modules: {list(act_scales.keys())[:10]}...")
        act_scale = weight.abs().mean(dim=0) * 0.1 + 0.01

    # Test different AWQ configurations
    configs = [
        {"group_size": 128, "name": "awq_gs128"},
        {"group_size": 64, "name": "awq_gs64"},
        {"group_size": 32, "name": "awq_gs32"},
        {"group_size": 256, "name": "awq_gs256"},
    ]

    results = []

    for cfg in configs:
        print(f"Testing {cfg['name']} (group_size={cfg['group_size']})...")

        dequantized, meta = quantize_awq_int4(
            weight,
            activation_scales=act_scale,
            group_size=cfg['group_size'],
        )

        metrics = compute_metrics(weight, dequantized)
        meta.update({
            "relative_frobenius_error": metrics.relative_frobenius_error,
            "max_absolute_error": metrics.max_absolute_error,
            "snr_db": metrics.snr_db,
            "compression_ratio": metrics.compression_ratio,
        })

        results.append(meta)
        print(f"  Error: {metrics.relative_frobenius_error*100:.2f}% " f"SNR: {metrics.snr_db:.2f}dB " f"Compress: {metrics.compression_ratio:.2f}x")

    # Compare with baseline INT4 group quantization
    print("Comparing with baseline INT4 group quantization...")
    dequant_baseline, scales = quantize_int4_group(weight, group_size=128)
    metrics_baseline = compute_metrics(weight, dequant_baseline)
    print(f"  Baseline INT4 group(128): Error={metrics_baseline.relative_frobenius_error*100:.2f}% " f"SNR={metrics_baseline.snr_db:.2f}dB")

    # Save results
    result_directory.mkdir(parents=True, exist_ok=True)
    with (result_directory / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(make_serializable({
            "model": MODEL_NAME,
            "parameter": PARAMETER_NAME,
            "calibration_samples": CALIBRATION_SAMPLES,
            "activation_scale_stats": {
                "mean": act_scale.mean().item(),
                "std": act_scale.std().item(),
                "min": act_scale.min().item(),
                "max": act_scale.max().item(),
            },
            "configs": results,
            "baseline_int4_group": {
                "error_pct": metrics_baseline.relative_frobenius_error * 100,
                "snr_db": metrics_baseline.snr_db,
                "compression_ratio": metrics_baseline.compression_ratio,
            }
        }), f, indent=2)

    print(f"Results saved to {result_directory / 'summary.json'}")


if __name__ == "__main__":
    main()
