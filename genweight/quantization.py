from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_weight_format(weight: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """
    Detect GPT-2 Conv1D weight format and transpose if needed.
    
    Standard linear: [out_features, in_features]
    GPT-2 Conv1D: [in_features, out_features]
    
    Returns: (possibly_transposed_weight, was_transposed)
    """
    rows, cols = weight.shape
    
    # Check if first dim is a "hidden size" and second is a multiple
    hidden_sizes = [768, 1024, 1280, 1536, 2048, 2560, 3072, 4096, 5120, 8192]
    for hs in hidden_sizes:
        if rows == hs and cols % hs == 0:
            # Likely Conv1D format [in_features, out_features]
            return weight.T, True
    
    # Check reverse
    for hs in hidden_sizes:
        if cols == hs and rows % hs == 0:
            # Standard linear format [out_features, in_features]
            return weight, False
    
    # Default: assume standard format
    return weight, False


def _restore_weight_format(weight: torch.Tensor, was_transposed: bool) -> torch.Tensor:
    """Restore original weight format."""
    return weight.T if was_transposed else weight


@dataclass
class QuantizationResult:
    """Results from a single quantization experiment."""
    scheme: str
    relative_frobenius_error: float
    max_absolute_error: float
    snr_db: float
    compression_ratio: float


# ============================================================
# EXISTING BASIC QUANTIZATION FUNCTIONS (kept for compatibility)
# ============================================================

def quantize_int8(tensor: torch.Tensor, scale: float | None = None) -> tuple[torch.Tensor, float]:
    """Simulate symmetric INT8 quantization with per-tensor scale."""
    weight, was_transposed = _get_weight_format(tensor)
    if scale is None:
        scale = weight.abs().max().item() / 127.0
    if scale == 0:
        scale = 1.0
    quantized = (weight / scale).round().clamp(-128, 127)
    dequantized = quantized * scale
    dequantized = _restore_weight_format(dequantized, was_transposed)
    return dequantized, scale


def quantize_int4(tensor: torch.Tensor, scale: float | None = None) -> tuple[torch.Tensor, float]:
    """Simulate symmetric INT4 quantization with per-tensor scale."""
    weight, was_transposed = _get_weight_format(tensor)
    if scale is None:
        scale = weight.abs().max().item() / 7.0
    if scale == 0:
        scale = 1.0
    quantized = (weight / scale).round().clamp(-8, 7)
    dequantized = quantized * scale
    dequantized = _restore_weight_format(dequantized, was_transposed)
    return dequantized, scale


def quantize_int4_group(tensor: torch.Tensor, group_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Simulate INT4 with group-wise scaling (GGUF-style)."""
    weight, was_transposed = _get_weight_format(tensor)
    original_shape = weight.shape
    flat = weight.flatten()
    num_groups = (flat.numel() + group_size - 1) // group_size
    padded_len = num_groups * group_size
    padded = torch.nn.functional.pad(flat, (0, padded_len - flat.numel()))
    grouped = padded.reshape(num_groups, group_size)

    scales = grouped.abs().max(dim=1).values / 7.0
    scales = scales.clamp_min(1e-8)
    quantized = (grouped / scales.unsqueeze(1)).round().clamp(-8, 7)
    dequantized = quantized * scales.unsqueeze(1)

    dequantized = dequantized.flatten()[:flat.numel()].reshape(original_shape)
    dequantized = _restore_weight_format(dequantized, was_transposed)
    return dequantized, scales


def quantize_fp8(tensor: torch.Tensor) -> torch.Tensor:
    """Simulate E4M3 FP8 quantization (approximate with float8_e4m3fn)."""
    weight, was_transposed = _get_weight_format(tensor)
    try:
        fp8_tensor = weight.to(torch.float8_e4m3fn)
        dequantized = fp8_tensor.to(weight.dtype)
    except (RuntimeError, AttributeError):
        # Fallback: simulate with manual scaling to E4M3 range
        max_val = 226.0
        scale = weight.abs().max().item() / max_val
        if scale == 0:
            return tensor
        quantized = (weight / scale).clamp(-max_val, max_val)
        dequantized = (quantized * scale).to(weight.dtype)
    dequantized = _restore_weight_format(dequantized, was_transposed)
    return dequantized


def quantize_fp4(tensor: torch.Tensor) -> torch.Tensor:
    """Simulate NVFP4-style quantization (2-bit mantissa, 2-bit exponent)."""
    weight, was_transposed = _get_weight_format(tensor)
    original_shape = weight.shape
    flat = weight.flatten()
    group_size = 16
    num_groups = (flat.numel() + group_size - 1) // group_size
    padded_len = num_groups * group_size
    padded = torch.nn.functional.pad(flat, (0, padded_len - flat.numel()))
    grouped = padded.reshape(num_groups, group_size)

    scales = grouped.abs().max(dim=1).values / 6.0
    scales = scales.clamp_min(1e-8)

    quantized = (grouped / scales.unsqueeze(1)).round().clamp(-6, 6)
    dequantized = quantized * scales.unsqueeze(1)

    dequantized = dequantized.flatten()[:flat.numel()].reshape(original_shape)
    dequantized = _restore_weight_format(dequantized, was_transposed)
    return dequantized


def compute_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    original_bits: int = 32,
    effective_bits: int = 8,
) -> QuantizationResult:
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
        compression_ratio=float(original_bits) / effective_bits,
    )


# ============================================================
# SMART QUANTIZATION METHODS (GPTQ, AWQ, SmoothQuant, AutoRound, NF4, GGUF, NVFP4)
# ============================================================

def quantize_gptq_int4(
    weight: torch.Tensor,
    hessian_diag: Optional[torch.Tensor] = None,
    group_size: int = 128,
    dampening: float = 0.01,
) -> tuple[torch.Tensor, dict]:
    """
    GPTQ-style weight-only quantization using Hessian-based importance.

    GPTQ key insight: Not all weights are equally important.
    Uses diagonal Hessian (second derivative of loss w.r.t. weights) as importance metric.
    Weights with higher Hessian diagonal are quantized more carefully.

    Args:
        weight: 2D weight matrix [out_features, in_features]
        hessian_diag: Precomputed diagonal Hessian [out_features, in_features] or [in_features]
        group_size: Block size for quantization
        dampening: Dampening factor for numerical stability

    Returns:
        dequantized weight, metadata dict
    """
    weight, was_transposed = _get_weight_format(weight)
    device = weight.device
    out_features, in_features = weight.shape
    weight = weight.float()

    # If no Hessian provided, use ones (reduces to naive INT4)
    if hessian_diag is None:
        hessian_diag = torch.ones(in_features, device=device)
    elif hessian_diag.dim() == 2:
        # Average over output features if per-output Hessian
        hessian_diag = hessian_diag.mean(dim=0)

    # Process in groups along input dimension
    num_groups = (in_features + group_size - 1) // group_size
    dequantized = torch.zeros_like(weight)
    scales = torch.zeros(num_groups, device=device)
    quantized_groups = []

    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, in_features)
        group_weight = weight[:, start:end]  # [out_features, group_size]
        group_hessian = hessian_diag[start:end]  # [group_size]

        # GPTQ: compute importance-weighted scale
        # Scale inversely proportional to sqrt(Hessian) to protect important weights
        importance = 1.0 / torch.sqrt(group_hessian + dampening)
        importance = importance / importance.mean()  # Normalize

        # Weighted max for scale computation
        weighted_weight = group_weight * importance.unsqueeze(0)
        scale = weighted_weight.abs().max().item() / 7.0
        scale = max(scale, 1e-8)
        scales[g] = scale

        # Quantize
        quantized = (group_weight / scale).round().clamp(-8, 7)
        dequantized[:, start:end] = quantized * scale
        quantized_groups.append(quantized)

    dequantized = _restore_weight_format(dequantized, was_transposed)

    metadata = {
        "scheme": "gptq_int4",
        "group_size": group_size,
        "scales": scales.cpu(),
        "dampening": dampening,
    }
    return dequantized, metadata


def quantize_awq_int4(
    weight: torch.Tensor,
    activation_scales: Optional[torch.Tensor] = None,
    group_size: int = 128,
) -> tuple[torch.Tensor, dict]:
    """
    AWQ-style quantization: Activation-Aware Weight Quantization.

    AWQ key insight: Protect weights that correspond to large activations.
    Importance = activation_scale^2 (since error propagates as scale^2).

    Args:
        weight: 2D weight matrix [out_features, in_features]
        activation_scales: Per-input-channel activation scales [in_features]
                          (obtained from calibration data)
        group_size: Quantization group size

    Returns:
        dequantized weight, metadata dict
    """
    weight, was_transposed = _get_weight_format(weight)
    device = weight.device
    out_features, in_features = weight.shape
    weight = weight.float()

    if activation_scales is None:
        # Fallback: use weight magnitude as proxy
        activation_scales = weight.abs().mean(dim=0)

    num_groups = (in_features + group_size - 1) // group_size
    dequantized = torch.zeros_like(weight)
    scales = torch.zeros(num_groups, device=device)

    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, in_features)
        group_weight = weight[:, start:end]
        group_act_scales = activation_scales[start:end]

        # AWQ: scale inversely to activation magnitude
        # Larger activation -> smaller scale -> finer quantization
        importance = 1.0 / (group_act_scales + 1e-8)
        importance = importance / importance.mean()

        weighted_weight = group_weight * importance.unsqueeze(0)
        scale = weighted_weight.abs().max().item() / 7.0
        scale = max(scale, 1e-8)
        scales[g] = scale

        quantized = (group_weight / scale).round().clamp(-8, 7)
        dequantized[:, start:end] = quantized * scale

    dequantized = _restore_weight_format(dequantized, was_transposed)

    metadata = {
        "scheme": "awq_int4",
        "group_size": group_size,
        "scales": scales.cpu(),
    }
    return dequantized, metadata


def quantize_smoothquant(
    weight: torch.Tensor,
    activation_scales: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
    scheme: str = "int8",
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """
    SmoothQuant: Smooth activation outliers by migrating scale to weights.

    Key idea: W * x = (W * s) * (x / s) where s = max(x)^alpha
    This makes activations easier to quantize at cost of weight quantization.

    Args:
        weight: [out_features, in_features]
        activation_scales: Per-channel activation max [in_features]
        alpha: Migration strength (0 = no migration, 1 = full migration)
        scheme: Target quantization for weights ("int8" or "int4")

    Returns:
        smoothed_weight, smoothed_activation_scales, metadata
    """
    weight, was_transposed = _get_weight_format(weight)
    device = weight.device
    out_features, in_features = weight.shape
    weight = weight.float()

    if activation_scales is None:
        # Proxy: use row norms of weight
        activation_scales = weight.abs().mean(dim=0)

    # Smooth factor: s = max(act)^alpha / max(w)^(1-alpha)
    # Simplified: use activation scales directly with alpha
    smooth_scales = activation_scales ** alpha
    smooth_scales = smooth_scales / smooth_scales.mean()

    # Migrate scale to weights
    smoothed_weight = weight * smooth_scales.unsqueeze(0)
    smoothed_act_scales = activation_scales / smooth_scales

    # Quantize smoothed weights
    if scheme == "int8":
        dequantized, _ = quantize_int8(smoothed_weight)
    else:
        dequantized, _ = quantize_int4(smoothed_weight)

    dequantized = _restore_weight_format(dequantized, was_transposed)

    metadata = {
        "scheme": f"smoothquant_{scheme}",
        "alpha": alpha,
        "smooth_scales": smooth_scales.cpu(),
        "smoothed_activation_scales": smoothed_act_scales.cpu(),
        "smoothed_weight_stats": {
            "mean": smoothed_weight.mean().item(),
            "std": smoothed_weight.std().item(),
        }
    }
    return dequantized, smoothed_act_scales, metadata


def quantize_autoround_int4(
    weight: torch.Tensor,
    hessian_diag: Optional[torch.Tensor] = None,
    group_size: int = 128,
    num_iterations: int = 200,
    lr: float = 1e-2,
) -> tuple[torch.Tensor, dict]:
    """
    AutoRound-style quantization: Optimize rounding decisions via gradient descent.

    Instead of naive round-to-nearest, learn optimal integer assignments
    that minimize reconstruction error weighted by Hessian importance.

    Args:
        weight: 2D weight matrix
        hessian_diag: Diagonal Hessian for importance weighting
        group_size: Quantization group size
        num_iterations: Optimization steps
        lr: Learning rate for scale/offset optimization

    Returns:
        dequantized weight, metadata
    """
    weight, was_transposed = _get_weight_format(weight)
    device = weight.device
    out_features, in_features = weight.shape
    weight = weight.float()

    if hessian_diag is None:
        hessian_diag = torch.ones(in_features, device=device)
    elif hessian_diag.dim() == 2:
        hessian_diag = hessian_diag.mean(dim=0)

    num_groups = (in_features + group_size - 1) // group_size
    dequantized = torch.zeros_like(weight)
    scales = torch.zeros(num_groups, device=device)

    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, in_features)
        group_weight = weight[:, start:end]  # [out, g]
        group_hessian = hessian_diag[start:end]

        # Initialize scale with weighted max
        importance = 1.0 / torch.sqrt(group_hessian + 0.01)
        importance = importance / importance.mean()
        weighted = group_weight * importance.unsqueeze(0)
        init_scale = weighted.abs().max().item() / 7.0
        init_scale = max(init_scale, 1e-8)

        # Learnable scale
        scale = nn.Parameter(torch.tensor(init_scale, device=device))
        optimizer = torch.optim.Adam([scale], lr=lr)

        for _ in range(num_iterations):
            optimizer.zero_grad()
            # Quantize with current scale
            quantized = (group_weight / scale).round().clamp(-8, 7)
            dequant = quantized * scale
            # Weighted reconstruction error
            error = (dequant - group_weight) * importance.unsqueeze(0)
            loss = (error ** 2).sum()
            loss.backward()
            optimizer.step()
            scale.data.clamp_(min=1e-8)

        final_scale = scale.detach()
        scales[g] = final_scale
        quantized = (group_weight / final_scale).round().clamp(-8, 7)
        dequantized[:, start:end] = quantized * final_scale

    dequantized = _restore_weight_format(dequantized, was_transposed)

    metadata = {
        "scheme": "autoround_int4",
        "group_size": group_size,
        "scales": scales.cpu(),
        "iterations": num_iterations,
    }
    return dequantized, metadata


def quantize_nf4(
    weight: torch.Tensor,
    block_size: int = 64,
) -> tuple[torch.Tensor, dict]:
    """
    NF4 (NormalFloat4) quantization from QLoRA.

    NF4 places 16 quantization levels according to the normal distribution
    where LLM weights typically concentrate (near 0).

    Levels are computed once and hardcoded for standard normal distribution.
    """
    # NF4 levels for standard normal distribution N(0,1)
    # These are the optimal 16 values from Dettmers et al. 2023
    NF4_LEVELS = torch.tensor([
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495,
        0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
        0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
        0.7229568362236023, 1.0
    ], device=weight.device)

    weight, was_transposed = _get_weight_format(weight)
    original_shape = weight.shape
    flat = weight.flatten()
    num_blocks = (flat.numel() + block_size - 1) // block_size
    padded_len = num_blocks * block_size
    padded = torch.nn.functional.pad(flat, (0, padded_len - flat.numel()))
    grouped = padded.reshape(num_blocks, block_size)

    # Normalize each block to N(0,1) scale
    scales = grouped.abs().mean(dim=1) * 1.25  # Empirical scale factor
    scales = scales.clamp_min(1e-8)
    normalized = grouped / scales.unsqueeze(1)

    # Quantize to nearest NF4 level
    # Use broadcasting: [num_blocks, block_size, 1] - [16] -> [num_blocks, block_size, 16]
    diff = normalized.unsqueeze(-1) - NF4_LEVELS.unsqueeze(0).unsqueeze(0)
    indices = diff.abs().argmin(dim=-1)
    quantized = NF4_LEVELS[indices]

    # Dequantize
    dequantized = quantized * scales.unsqueeze(1)
    dequantized = dequantized.flatten()[:flat.numel()].reshape(original_shape)
    dequantized = _restore_weight_format(dequantized, was_transposed)

    metadata = {
        "scheme": "nf4",
        "block_size": block_size,
        "scales": scales.cpu(),
        "levels": NF4_LEVELS.cpu(),
    }
    return dequantized, metadata


def quantize_gguf_q4_k(
    weight: torch.Tensor,
    block_size: int = 256,
) -> tuple[torch.Tensor, dict]:
    """
    GGUF-style Q4_K_M quantization (simplified).

    Q4_K uses:
    - 4-bit values per weight
    - Per-block scale (16-bit float)
    - Per-block min/offset for better range utilization
    - Super-block structure for efficient packing

    This is a simplified functional version.
    """
    weight, was_transposed = _get_weight_format(weight)
    original_shape = weight.shape
    flat = weight.flatten()
    num_blocks = (flat.numel() + block_size - 1) // block_size
    padded_len = num_blocks * block_size
    padded = torch.nn.functional.pad(flat, (0, padded_len - flat.numel()))
    grouped = padded.reshape(num_blocks, block_size)

    # Q4_K uses 4-bit values: 0-15 mapped to symmetric range with offset
    scales = torch.zeros(num_blocks, device=weight.device)
    mins = torch.zeros(num_blocks, device=weight.device)
    dequantized = torch.zeros_like(grouped)

    for b in range(num_blocks):
        block = grouped[b]
        block_min = block.min()
        block_max = block.max()

        # Use symmetric scale around midpoint for better utilization
        scale = (block_max - block_min) / 15.0
        scale = max(scale, 1e-8)

        # Quantize to 0-15
        quantized = ((block - block_min) / scale).round().clamp(0, 15).to(torch.uint8)
        # Dequantize
        dequantized[b] = quantized.float() * scale + block_min

        scales[b] = scale
        mins[b] = block_min

    dequantized = dequantized.flatten()[:flat.numel()].reshape(original_shape)
    dequantized = _restore_weight_format(dequantized, was_transposed)

    metadata = {
        "scheme": "gguf_q4_k",
        "block_size": block_size,
        "scales": scales.cpu(),
        "mins": mins.cpu(),
    }
    return dequantized, metadata


def quantize_nvfp4(
    weight: torch.Tensor,
    group_size: int = 16,
    scale_precision: str = "fp8",
) -> tuple[torch.Tensor, dict]:
    """
    NVFP4 microscaling (Blackwell architecture style).

    Key innovations:
    1. Ultra-fine groups (16 weights share one scale)
    2. Scale stored in higher precision (FP8 or FP16)
    3. 4-bit values are floating-point (1 sign, 2 exp, 1 mantissa)

    This is a functional simulation of the format.
    """
    weight, was_transposed = _get_weight_format(weight)
    original_shape = weight.shape
    flat = weight.flatten()
    num_groups = (flat.numel() + group_size - 1) // group_size
    padded_len = num_groups * group_size
    padded = torch.nn.functional.pad(flat, (0, padded_len - flat.numel()))
    grouped = padded.reshape(num_groups, group_size)

    # NVFP4 representable values (E2M1 format: 1 sign, 2 exponent, 1 mantissa)
    # Values: 0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
    NVFP4_VALUES = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    ], device=weight.device)

    scales = torch.zeros(num_groups, device=weight.device)
    dequantized = torch.zeros_like(grouped)

    for g in range(num_groups):
        group = grouped[g]

        # Find optimal scale for this group
        max_abs = group.abs().max()
        if max_abs > 0:
            scale = max_abs / 6.0  # 6.0 is max NVFP4 value
        else:
            scale = 1.0
        scales[g] = scale

        # Normalize and quantize to nearest NVFP4 value
        normalized = group / scale
        diff = normalized.unsqueeze(-1) - NVFP4_VALUES.unsqueeze(0)
        indices = diff.abs().argmin(dim=-1)
        quantized = NVFP4_VALUES[indices]
        dequantized[g] = quantized * scale

    dequantized = dequantized.flatten()[:flat.numel()].reshape(original_shape)
    dequantized = _restore_weight_format(dequantized, was_transposed)

    metadata = {
        "scheme": "nvfp4",
        "group_size": group_size,
        "scales": scales.cpu(),
        "scale_precision": scale_precision,
    }
    return dequantized, metadata


# ============================================================
# METRICS COMPUTATION
# ============================================================

def compute_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    original_bits: int = 32,
    effective_bits: int = 8,
) -> QuantizationResult:
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
        compression_ratio=float(original_bits) / effective_bits,
    )


# ============================================================
# HIGH-LEVEL INTERFACE
# ============================================================

BASIC_QUANTIZATION_SCHEMES = [
    "int8",
    "int8_per_channel",
    "int4",
    "int4_per_channel",
    "int4_group",
    "fp8",
    "fp4",
]

SMART_QUANTIZATION_SCHEMES = [
    "gptq_int4",
    "awq_int4",
    "smoothquant_int8",
    "smoothquant_int4",
    "autoround_int4",
    "nf4",
    "gguf_q4_k",
    "nvfp4",
]

ALL_QUANTIZATION_SCHEMES = BASIC_QUANTIZATION_SCHEMES + SMART_QUANTIZATION_SCHEMES


def quantize_matrix_basic(
    tensor: torch.Tensor,
    scheme: str,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """Quantize a matrix with basic schemes (delegates to old implementation)."""
    # This is a placeholder - in practice use quantize_matrix_smart
    # which handles all schemes uniformly
    if scheme in BASIC_QUANTIZATION_SCHEMES:
        # Call appropriate basic function
        if scheme == "int8":
            dequant, scale = quantize_int8(tensor)
            metrics = compute_metrics(tensor, dequant)
            return dequant, {
                "scheme": f"int8_per_tensor(scale={scale:.6f})",
                "relative_frobenius_error": metrics.relative_frobenius_error,
                "max_absolute_error": metrics.max_absolute_error,
                "snr_db": metrics.snr_db,
                "compression_ratio": metrics.compression_ratio,
            }
        elif scheme == "int4":
            dequant, scale = quantize_int4(tensor)
            metrics = compute_metrics(tensor, dequant)
            return dequant, {
                "scheme": f"int4_per_tensor(scale={scale:.6f})",
                "relative_frobenius_error": metrics.relative_frobenius_error,
                "max_absolute_error": metrics.max_absolute_error,
                "snr_db": metrics.snr_db,
                "compression_ratio": metrics.compression_ratio,
            }
        elif scheme == "int4_group":
            dequant, scales = quantize_int4_group(tensor, kwargs.get("group_size", 64))
            metrics = compute_metrics(tensor, dequant)
            return dequant, {
                "scheme": f"int4_group(group_size={kwargs.get('group_size', 64)})",
                "relative_frobenius_error": metrics.relative_frobenius_error,
                "max_absolute_error": metrics.max_absolute_error,
                "snr_db": metrics.snr_db,
                "compression_ratio": metrics.compression_ratio,
            }
        elif scheme == "fp8":
            dequant = quantize_fp8(tensor)
            metrics = compute_metrics(tensor, dequant, effective_bits=8)
            return dequant, {
                "scheme": "fp8_e4m3",
                "relative_frobenius_error": metrics.relative_frobenius_error,
                "max_absolute_error": metrics.max_absolute_error,
                "snr_db": metrics.snr_db,
                "compression_ratio": metrics.compression_ratio,
            }
        elif scheme == "fp4":
            dequant = quantize_fp4(tensor)
            metrics = compute_metrics(tensor, dequant, effective_bits=4)
            return dequant, {
                "scheme": "fp4_nvfp4_style",
                "relative_frobenius_error": metrics.relative_frobenius_error,
                "max_absolute_error": metrics.max_absolute_error,
                "snr_db": metrics.snr_db,
                "compression_ratio": metrics.compression_ratio,
            }
    
    raise ValueError(f"Unknown basic scheme: {scheme}")


def quantize_matrix_smart(
    tensor: torch.Tensor,
    scheme: str,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """Unified interface for all quantization schemes."""
    if scheme in BASIC_QUANTIZATION_SCHEMES:
        return quantize_matrix_basic(tensor, scheme, **kwargs)

    # Smart quantization schemes
    if scheme == "gptq_int4":
        hessian = kwargs.get("hessian_diag", None)
        dequant, meta = quantize_gptq_int4(tensor, hessian_diag=hessian,
                                            group_size=kwargs.get("group_size", 128))
    elif scheme == "awq_int4":
        act_scales = kwargs.get("activation_scales", None)
        dequant, meta = quantize_awq_int4(tensor, activation_scales=act_scales,
                                           group_size=kwargs.get("group_size", 128))
    elif scheme in ["smoothquant_int8", "smoothquant_int4"]:
        act_scales = kwargs.get("activation_scales", None)
        alpha = kwargs.get("alpha", 0.5)
        target = "int8" if "int8" in scheme else "int4"
        dequant, smoothed_act, meta = quantize_smoothquant(tensor, act_scales, alpha, target)
    elif scheme == "autoround_int4":
        hessian = kwargs.get("hessian_diag", None)
        dequant, meta = quantize_autoround_int4(tensor, hessian_diag=hessian,
                                                 group_size=kwargs.get("group_size", 128),
                                                 num_iterations=kwargs.get("iterations", 200))
    elif scheme == "nf4":
        dequant, meta = quantize_nf4(tensor, block_size=kwargs.get("block_size", 64))
    elif scheme == "gguf_q4_k":
        dequant, meta = quantize_gguf_q4_k(tensor, block_size=kwargs.get("block_size", 256))
    elif scheme == "nvfp4":
        dequant, meta = quantize_nvfp4(tensor, group_size=kwargs.get("group_size", 16))
    else:
        raise ValueError(f"Unknown quantization scheme: {scheme}")

    # Compute metrics
    metrics = compute_metrics(tensor, dequant)
    meta.update({
        "relative_frobenius_error": metrics.relative_frobenius_error,
        "max_absolute_error": metrics.max_absolute_error,
        "snr_db": metrics.snr_db,
        "compression_ratio": metrics.compression_ratio,
    })
    return dequant, meta


def run_full_quantization_suite(
    tensor: torch.Tensor,
    schemes: list[str] | None = None,
    hessian_diag: Optional[torch.Tensor] = None,
    activation_scales: Optional[torch.Tensor] = None,
    **kwargs,
) -> list[dict]:
    """Run all quantization schemes on a tensor."""
    if schemes is None:
        schemes = ALL_QUANTIZATION_SCHEMES

    results = []
    for scheme in schemes:
        try:
            dequant, meta = quantize_matrix_smart(
                tensor, scheme,
                hessian_diag=hessian_diag,
                activation_scales=activation_scales,
                **kwargs,
            )
            meta["scheme"] = scheme
            results.append(meta)
        except Exception as e:
            results.append({
                "scheme": f"{scheme} (ERROR: {e})",
                "relative_frobenius_error": float('inf'),
                "max_absolute_error": float('inf'),
                "snr_db": -float('inf'),
                "compression_ratio": 0.0,
            })
    return results
