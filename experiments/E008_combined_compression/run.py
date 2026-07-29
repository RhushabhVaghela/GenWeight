"""E008 — Combined compression pipeline: QK reuse + Low-rank + Quantization.

Stacks multiple compression techniques and measures the total compression ratio
vs. total reconstruction error, comparing against individual methods.
"""

import json
from pathlib import Path

import torch

from genweight import (
    ModelLoader,
    QKVSimilarityAnalyzer,
    LowRankAnalyzer,
    run_quantization_suite,
)


MODEL_NAME = "gpt2"
PARAMETER_NAME = "h.0.attn.c_attn.weight"
BLOCK_SIZE = 64


def compress_qk_reuse(weight: torch.Tensor, similarity_threshold: float = 0.75, residual_rank: int = 16) -> tuple[torch.Tensor, dict]:
    """Apply Q→K linear reuse with low-rank residual.

    Returns reconstructed K segment and metadata.
    """
    rows, cols = weight.shape
    segment_width = cols // 3
    head_count = segment_width // BLOCK_SIZE

    segments = weight.reshape(rows, 3, segment_width).permute(1, 0, 2)
    q_seg = segments[0]  # (rows, segment_width)
    k_seg = segments[1]
    v_seg = segments[2]

    q_heads = q_seg.reshape(rows, head_count, BLOCK_SIZE).permute(1, 0, 2)
    k_heads = k_seg.reshape(rows, head_count, BLOCK_SIZE).permute(1, 0, 2)

    selected_heads = []
    transforms = []
    residuals = []

    for h in range(head_count):
        q = q_heads[h]
        k = k_heads[h]
        cosine = (torch.sum(q * k) / (torch.linalg.vector_norm(q) * torch.linalg.vector_norm(k))).item()

        if cosine >= similarity_threshold:
            A = torch.linalg.lstsq(q, k).solution
            r = k - q @ A
            selected_heads.append(h)
            transforms.append(A)
            residuals.append(r)

    # Reconstruct K segment
    k_recon = torch.zeros_like(k_seg)
    k_recon = k_recon.reshape(rows, head_count, BLOCK_SIZE).permute(1, 0, 2)

    for idx, h in enumerate(selected_heads):
        k_recon[h] = q_heads[h] @ transforms[idx]
        if residual_rank > 0 and residuals[idx].numel() > 0:
            U, S, Vh = torch.linalg.svd(residuals[idx], full_matrices=False)
            r = min(residual_rank, len(S))
            k_recon[h] += (U[:, :r] * S[:r]) @ Vh[:r, :]

    k_recon = k_recon.permute(1, 0, 2).reshape(rows, segment_width)

    # Unchanged Q and V
    q_recon = q_seg
    v_recon = v_seg

    full_recon = torch.cat([q_recon, k_recon, v_recon], dim=1)

    dense_params = weight.numel()
    qk_dense = rows * segment_width * 2  # Q + V stored
    selected_count = len(selected_heads)
    head_params = rows * BLOCK_SIZE
    transform_params = selected_count * BLOCK_SIZE * BLOCK_SIZE
    residual_params = selected_count * residual_rank * (rows + BLOCK_SIZE)
    compressed_params = qk_dense + transform_params + residual_params

    return full_recon, {
        "selected_heads": selected_heads,
        "head_count": selected_count,
        "compression_ratio": dense_params / compressed_params,
        "compressed_params": compressed_params,
    }


def compress_low_rank(weight: torch.Tensor, rank: int) -> tuple[torch.Tensor, dict]:
    """Low-rank SVD reconstruction."""
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    r = min(rank, len(S))
    recon = (U[:, :r] * S[:r]) @ Vh[:r, :]

    dense_params = weight.numel()
    factor_params = r * (weight.shape[0] + weight.shape[1] + 1)

    return recon, {
        "rank": r,
        "compression_ratio": dense_params / factor_params,
        "compressed_params": factor_params,
    }


def compress_quantize(weight: torch.Tensor, scheme: str) -> tuple[torch.Tensor, dict]:
    """Quantize weight matrix."""
    from genweight.quantization import quantize_matrix
    dequantized, result = quantize_matrix(weight, scheme)
    return dequantized, {
        "scheme": result.scheme,
        "compression_ratio": result.compression_ratio,
        "error_pct": result.relative_frobenius_error * 100,
    }


def main() -> None:
    result_directory = Path("results/E008_combined_compression")
    loader = ModelLoader(MODEL_NAME)
    loader.load()
    weight = loader.get_parameter(PARAMETER_NAME)

    dense_params = weight.numel()
    print(f"Original: {tuple(weight.shape)} = {dense_params:,} params")

    # --- Individual methods ---
    print("\n" + "=" * 60)
    print("Individual compression methods")
    print("=" * 60)

    # 1. QK reuse only
    recon_qk, meta_qk = compress_qk_reuse(weight, similarity_threshold=0.75, residual_rank=16)
    err_qk = torch.linalg.vector_norm(recon_qk - weight) / torch.linalg.vector_norm(weight)
    print(f"QK-reuse (thresh=0.75, res_rank=16):  err={err_qk*100:.2f}%  comp={meta_qk['compression_ratio']:.2f}x  params={meta_qk['compressed_params']:,}")

    # 2. Low-rank only (at energy_rank_95 = 501)
    recon_lr, meta_lr = compress_low_rank(weight, rank=501)
    err_lr = torch.linalg.vector_norm(recon_lr - weight) / torch.linalg.vector_norm(weight)
    print(f"Low-rank (rank=501, 95% energy):     err={err_lr*100:.2f}%  comp={meta_lr['compression_ratio']:.2f}x  params={meta_lr['compressed_params']:,}")

    # 3. INT8 per-channel quantization
    recon_q8, meta_q8 = compress_quantize(weight, "int8_per_channel")
    err_q8 = torch.linalg.vector_norm(recon_q8 - weight) / torch.linalg.vector_norm(weight)
    print(f"INT8 per-channel:                    err={err_q8*100:.2f}%  comp={meta_q8['compression_ratio']:.2f}x")

    # 4. INT4 group quantization
    recon_q4, meta_q4 = compress_quantize(weight, "int4_group")
    err_q4 = torch.linalg.vector_norm(recon_q4 - weight) / torch.linalg.vector_norm(weight)
    print(f"INT4 group (64):                     err={err_q4*100:.2f}%  comp={meta_q4['compression_ratio']:.2f}x")

    # 5. FP8 quantization
    recon_fp8, meta_fp8 = compress_quantize(weight, "fp8")
    err_fp8 = torch.linalg.vector_norm(recon_fp8 - weight) / torch.linalg.vector_norm(weight)
    print(f"FP8:                                  err={err_fp8*100:.2f}%  comp={meta_fp8['compression_ratio']:.2f}x")

    # --- Combined pipelines ---
    print("\n" + "=" * 60)
    print("Combined compression pipelines")
    print("=" * 60)

    # Pipeline A: QK-reuse → Quantize the residual
    print("\nPipeline A: QK-reuse + INT4 group on residual")
    # QK-reuse reconstructs K, leaving Q and V dense
    # Then quantize the whole thing with INT4
    recon_a = recon_qk.clone()
    # Actually let's quantize the full reconstructed matrix
    recon_a_q, meta_a = compress_quantize(recon_qk, "int4_group")
    err_a = torch.linalg.vector_norm(recon_a_q - weight) / torch.linalg.vector_norm(weight)
    total_comp_a = meta_qk["compression_ratio"] * meta_a["compression_ratio"]
    print(f"  QK-reuse → INT4:  err={err_a*100:.2f}%  comp={total_comp_a:.2f}x")

    # Pipeline B: Low-rank → Quantize
    print("\nPipeline B: Low-rank (rank=256) → INT4 group")
    recon_b_lr, meta_b_lr = compress_low_rank(weight, rank=256)
    recon_b_q, meta_b_q = compress_quantize(recon_b_lr, "int4_group")
    err_b = torch.linalg.vector_norm(recon_b_q - weight) / torch.linalg.vector_norm(weight)
    total_comp_b = meta_b_lr["compression_ratio"] * meta_b_q["compression_ratio"]
    print(f"  Low-rank(256) → INT4:  err={err_b*100:.2f}%  comp={total_comp_b:.2f}x")

    # Pipeline C: Low-rank → FP8
    print("\nPipeline C: Low-rank (rank=256) → FP8")
    recon_c_q, meta_c_q = compress_quantize(recon_b_lr, "fp8")
    err_c = torch.linalg.vector_norm(recon_c_q - weight) / torch.linalg.vector_norm(weight)
    total_comp_c = meta_b_lr["compression_ratio"] * meta_c_q["compression_ratio"]
    print(f"  Low-rank(256) → FP8:  err={err_c*100:.2f}%  comp={total_comp_c:.2f}x")

    # Pipeline D: QK-reuse + Low-rank residual on full matrix
    print("\nPipeline D: QK-reuse on attn + Low-rank on residual of full matrix")
    # First apply QK reuse to get partial reconstruction
    # Then compute residual on full matrix and apply low-rank
    residual_full = weight - recon_qk
    # Low-rank on residual
    U, S, Vh = torch.linalg.svd(residual_full, full_matrices=False)
    r = 128
    residual_lr = (U[:, :r] * S[:r]) @ Vh[:r, :]
    recon_d = recon_qk + residual_lr
    err_d = torch.linalg.vector_norm(recon_d - weight) / torch.linalg.vector_norm(weight)
    params_d = meta_qk["compressed_params"] + r * (weight.shape[0] + weight.shape[1] + 1)
    comp_d = dense_params / params_d
    print(f"  QK-reuse + LR(128) on residual:  err={err_d*100:.2f}%  comp={comp_d:.2f}x")

    # Pipeline E: Quantize → Low-rank on quantized (less typical)
    print("\nPipeline E: INT8 per-channel → Low-rank on quantized")
    recon_e_lr, meta_e_lr = compress_low_rank(recon_q8, rank=256)
    err_e = torch.linalg.vector_norm(recon_e_lr - weight) / torch.linalg.vector_norm(weight)
    comp_e = meta_q8["compression_ratio"] * (dense_params / meta_e_lr["compressed_params"])
    print(f"  INT8 → LR(256):  err={err_e*100:.2f}%  comp={comp_e:.2f}x")

    # Save summary
    result_directory.mkdir(parents=True, exist_ok=True)
    summary = {
        "original_params": dense_params,
        "individual": {
            "qk_reuse": {"error_pct": err_qk.item() * 100, "compression_ratio": meta_qk["compression_ratio"]},
            "low_rank_501": {"error_pct": err_lr.item() * 100, "compression_ratio": meta_lr["compression_ratio"]},
            "low_rank_256": {"error_pct": (torch.linalg.vector_norm(compress_low_rank(weight, 256)[0] - weight) / torch.linalg.vector_norm(weight)).item() * 100, "compression_ratio": dense_params / (256 * (weight.shape[0] + weight.shape[1] + 1))},
            "int8_per_channel": {"error_pct": err_q8.item() * 100, "compression_ratio": meta_q8["compression_ratio"]},
            "int4_group": {"error_pct": err_q4.item() * 100, "compression_ratio": meta_q4["compression_ratio"]},
            "fp8": {"error_pct": err_fp8.item() * 100, "compression_ratio": meta_fp8["compression_ratio"]},
        },
        "pipelines": {
            "A_qk_int4": {"error_pct": err_a.item() * 100, "compression_ratio": total_comp_a},
            "B_lr256_int4": {"error_pct": err_b.item() * 100, "compression_ratio": total_comp_b},
            "C_lr256_fp8": {"error_pct": err_c.item() * 100, "compression_ratio": total_comp_c},
            "D_qk_lr128_residual": {"error_pct": err_d.item() * 100, "compression_ratio": comp_d},
            "E_int8_lr256": {"error_pct": err_e.item() * 100, "compression_ratio": comp_e},
        },
    }
    with (result_directory / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {result_directory / 'summary.json'}")


if __name__ == "__main__":
    main()