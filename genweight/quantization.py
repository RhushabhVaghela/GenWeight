from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class QuantizationResult:
    """Results from a single quantization experiment."""
    scheme: str
    relative_frobenius_error: float
    max_absolute_error: float
    snr_db: float
    compression_ratio: float


def quantize_int8(tensor: torch.Tensor, scale: float | None = None) -> tuple[torch.Tensor, float]:
    """Simulate symmetric INT8 quantization with per-tensor scale."""
    if scale is None:
        scale = tensor.abs().max().item() / 127.0
    if scale == 0:
        scale = 1.0
    quantized = (tensor / scale).round().clamp(-128, 127)
    dequantized = quantized * scale
    return dequantized, scale


def quantize_int4(tensor: torch.Tensor, scale: float | None = None) -> tuple[torch.Tensor, float]:
    """Simulate symmetric INT4 quantization with per-tensor scale."""
    if scale is None:
        scale = tensor.abs().max().item() / 7.0
    if scale == 0:
        scale = 1.0
    quantized = (tensor / scale).round().clamp(-8, 7)
    dequantized = quantized * scale
    return dequantized, scale


def quantize_int4_group(tensor: torch.Tensor, group_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate INT4 with group-wise scaling (GGUF-style).

    Args:
        tensor: 2D weight matrix
        group_size: Number of elements per scaling group (along flattened dim)

    Returns:
        dequantized tensor, group scales
    """
    original_shape = tensor.shape
    flat = tensor.flatten()
    num_groups = (flat.numel() + group_size - 1) // group_size
    padded_len = num_groups * group_size
    padded = torch.nn.functional.pad(flat, (0, padded_len - flat.numel()))
    grouped = padded.reshape(num_groups, group_size)

    scales = grouped.abs().max(dim=1).values / 7.0
    scales = scales.clamp_min(1e-8)
    quantized = (grouped / scales.unsqueeze(1)).round().clamp(-8, 7)
    dequantized = quantized * scales.unsqueeze(1)

    dequantized = dequantized.flatten()[:flat.numel()].reshape(original_shape)
    return dequantized, scales


def quantize_fp8(tensor: torch.Tensor) -> torch.Tensor:
    """Simulate E4M3 FP8 quantization (approximate with float8_e4m3fn)."""
    # PyTorch 2.1+ has native float8 support
    try:
        fp8_tensor = tensor.to(torch.float8_e4m3fn)
        return fp8_tensor.to(tensor.dtype)
    except (RuntimeError, AttributeError):
        # Fallback: simulate with manual scaling to E4M3 range
        # E4M3 max value ≈ 240.0 (actually 226.0 for E4M3fn)
        max_val = 226.0
        scale = tensor.abs().max().item() / max_val
        if scale == 0:
            return tensor
        quantized = (tensor / scale).clamp(-max_val, max_val)
        # Round to nearest representable value (simplified)
        return (quantized * scale).to(tensor.dtype)


def quantize_fp4(tensor: torch.Tensor) -> torch.Tensor:
    """Simulate NVFP4-style quantization (2-bit mantissa, 2-bit exponent).

    This is a rough approximation of NVIDIA's NVFP4 format.
    """
    # NVFP4 has 16 values: {0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}
    # We'll use a simple power-of-2 scale per group of 16
    original_shape = tensor.shape
    flat = tensor.flatten()
    group_size = 16
    num_groups = (flat.numel() + group_size - 1) // group_size
    padded_len = num_groups * group_size
    padded = torch.nn.functional.pad(flat, (0, padded_len - flat.numel()))
    grouped = padded.reshape(num_groups, group_size)

    scales = grouped.abs().max(dim=1).values / 6.0  # 6.0 is max NVFP4
    scales = scales.clamp_min(1e-8)

    quantized = (grouped / scales.unsqueeze(1)).round().clamp(-6, 6)
    dequantized = quantized * scales.unsqueeze(1)

    dequantized = dequantized.flatten()[:flat.numel()].reshape(original_shape)
    return dequantized


def compute_metrics(original: torch.Tensor, reconstructed: torch.Tensor, original_bits: int = 32) -> QuantizationResult:
    """Compute quality metrics for quantization."""
    error = reconstructed - original
    fro_norm_original = torch.linalg.matrix_norm(original).item()
    fro_norm_error = torch.linalg.matrix_norm(error).item()
    relative_fro_error = fro_norm_error / max(fro_norm_original, 1e-12)
    max_abs_error = error.abs().max().item()

    # SNR in dB
    signal_power = (original ** 2).mean().item()
    noise_power = (error ** 2).mean().item()
    snr_db = 10 * torch.log10(torch.tensor(signal_power / max(noise_power, 1e-12))).item()

    return QuantizationResult(
        scheme="",
        relative_frobenius_error=relative_fro_error,
        max_absolute_error=max_abs_error,
        snr_db=snr_db,
        compression_ratio=float(original_bits) / 8.0,  # placeholder
    )


def quantize_matrix(tensor: torch.Tensor, scheme: str, **kwargs) -> tuple[torch.Tensor, QuantizationResult]:
    """Quantize a matrix with the specified scheme and return metrics."""
    original_bits = 32  # FP32 baseline

    if scheme == "int8":
        dequantized, scale = quantize_int8(tensor)
        result = compute_metrics(tensor, dequantized, original_bits)
        result.scheme = f"int8_per_tensor(scale={scale:.6f})"
        result.compression_ratio = original_bits / 8.0
    elif scheme == "int4":
        dequantized, scale = quantize_int4(tensor)
        result = compute_metrics(tensor, dequantized, original_bits)
        result.scheme = f"int4_per_tensor(scale={scale:.6f})"
        result.compression_ratio = original_bits / 4.0
    elif scheme == "int4_group":
        group_size = kwargs.get("group_size", 64)
        dequantized, scales = quantize_int4_group(tensor, group_size)
        result = compute_metrics(tensor, dequantized, original_bits)
        result.scheme = f"int4_group(group_size={group_size})"
        result.compression_ratio = original_bits / 4.0
    elif scheme == "fp8":
        dequantized = quantize_fp8(tensor)
        result = compute_metrics(tensor, dequantized, original_bits)
        result.scheme = "fp8_e4m3"
        result.compression_ratio = original_bits / 8.0
    elif scheme == "fp4":
        dequantized = quantize_fp4(tensor)
        result = compute_metrics(tensor, dequantized, original_bits)
        result.scheme = "fp4_nvfp4_style"
        result.compression_ratio = original_bits / 4.0
    elif scheme == "int8_per_channel":
        # Per-output-channel scaling (common for linear layers)
        dequantized = torch.zeros_like(tensor)
        scales = []
        for i in range(tensor.shape[0]):
            dq, scale = quantize_int8(tensor[i:i+1])
            dequantized[i:i+1] = dq
            scales.append(scale)
        result = compute_metrics(tensor, dequantized, original_bits)
        result.scheme = f"int8_per_channel({len(scales)} channels)"
        result.compression_ratio = original_bits / 8.0
    elif scheme == "int4_per_channel":
        dequantized = torch.zeros_like(tensor)
        for i in range(tensor.shape[0]):
            dq, _ = quantize_int4(tensor[i:i+1])
            dequantized[i:i+1] = dq
        result = compute_metrics(tensor, dequantized, original_bits)
        result.scheme = f"int4_per_channel({tensor.shape[0]} channels)"
        result.compression_ratio = original_bits / 4.0
    else:
        raise ValueError(f"Unknown quantization scheme: {scheme}")

    return dequantized, result


# Convenience: all supported schemes
QUANTIZATION_SCHEMES = [
    "int8",
    "int8_per_channel",
    "int4",
    "int4_per_channel",
    "int4_group",
    "fp8",
    "fp4",
]


def run_quantization_suite(
    tensor: torch.Tensor,
    schemes: list[str] | None = None,
    group_size: int = 64,
) -> list[QuantizationResult]:
    """Run quantization experiments on a single tensor."""
    if schemes is None:
        schemes = QUANTIZATION_SCHEMES

    results = []
    for scheme in schemes:
        try:
            _, result = quantize_matrix(tensor, scheme, group_size=group_size)
            results.append(result)
        except Exception as e:
            results.append(QuantizationResult(
                scheme=f"{scheme} (ERROR: {e})",
                relative_frobenius_error=float('inf'),
                max_absolute_error=float('inf'),
                snr_db=-float('inf'),
                compression_ratio=0.0,
            ))
    return results