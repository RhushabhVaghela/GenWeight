# GenWeight: Transformer Weight Geometry & Compression Research

> **Research framework for studying the geometric structure of learned transformer weights and their compressibility.**

---

## Project Overview

GenWeight systematically analyzes GPT-2's weight matrices to answer:

1. **Do learned weights have exploitable structure?** (vs. random Gaussian matrices)
2. **Does this structure generalize across layers?**
3. **What compression methods work best, and can we combine them?**

---

## Key Findings (Experiments E000–E008)

### E000 — Single-Layer Deep Dive (Layer 0, `h.0.attn.c_attn.weight`)

| Property | Value | Gaussian Baseline | Difference |
|----------|-------|-------------------|------------|
| Effective rank | **394** | 650 | **−256** (much lower rank) |
| Condition number | **45.5** | 3.7 | **+42×** (highly anisotropic) |
| Energy rank 90% | **399** | 559 | **−160** |
| Energy rank 95% | **501** | 641 | **−140** |

**→ The weight matrix is significantly more structured (low-rank, ill-conditioned) than random.**

---

### E006 — Multi-Layer Analysis (All 12 Attention Layers)

| Layer | Eff. Rank | Cond. # | Std | Q/K Mean Cosine | Q/K Heads >0.75 |
|-------|-----------|---------|-----|-----------------|-----------------|
| h.0   | 394.3 | 45.5 | 0.200 | **+0.172** | **3** |
| h.1   | 409.1 | 41.4 | 0.140 | −0.045 | 0 |
| h.2   | 454.6 | 50.3 | 0.153 | −0.229 | 0 |
| h.3   | 495.7 | 23.6 | 0.142 | −0.127 | 0 |
| h.4   | 474.2 | 28.4 | 0.146 | −0.024 | 0 |
| h.5   | 459.0 | 32.7 | 0.128 | +0.054 | 0 |
| h.6   | 494.7 | 12.9 | 0.127 | −0.001 | 0 |
| h.7   | 502.3 | 22.2 | 0.129 | +0.071 | 0 |
| h.8   | 549.4 | 9.7  | 0.127 | +0.060 | 0 |
| h.9   | 577.1 | 11.0 | 0.126 | +0.094 | 0 |
| h.10  | 545.1 | 9.7  | 0.127 | +0.111 | 0 |
| h.11  | 525.4 | 21.4 | 0.128 | +0.082 | 0 |

**Key observations:**
- **Effective rank increases with depth** (394 → 577): deeper layers are *less* compressible by low-rank approximation
- **Condition number decreases** (45.5 → 9.7): deeper layers are *better conditioned*
- **Q/K alignment is layer-specific**: Only layer 0 shows strong same-index Q→K reuse (3 heads > 0.75 cosine); other layers show near-zero or negative mean cosine
- **Weight std decreases** (0.20 → 0.126): weights become more concentrated in later layers

---

### E007 — Quantization Sensitivity (All 48 2D Weight Matrices)

| Scheme | Avg Error | Min Error | Max Error | Compression | Notes |
|--------|-----------|-----------|-----------|-------------|-------|
| **INT8 per-tensor** | ~4.5% | 2.0% | 42% | 4× | Poor on large output layers |
| **INT8 per-channel** | **~1.3%** | 0.9% | 2.9% | 4× | **Excellent** — consistently < 3% error |
| **INT4 per-tensor** | ~65% | 35% | 98% | 8× | Unusable |
| **INT4 per-channel** | ~22% | 16% | 41% | 8× | Still high error |
| **INT4 group (64)** | **~11%** | 9.7% | 17% | 8× | **Best INT4** — group quantization works well |
| **FP8 (E4M3)** | **~2.65%** | 2.6% | 2.7% | 4× | **Consistently excellent** — layer-independent |
| **FP4 (NVFP4-style)** | **~10.2%** | 7% | 12% | 8× | Good for 8-bit budget |

**Critical finding: INT8 per-channel quantization achieves ~1.3% error universally** — the most practical compression method. FP8 is even more consistent (~2.65% error). Group INT4 (~11%) is viable for extreme compression.

**Layer sensitivity varies:** MLP `c_proj` (3072→768) and embeddings are hardest for per-tensor quantization due to large dynamic range.

---

### E008 — Combined Compression Pipelines (Layer 0 `c_attn`)

| Pipeline | Error | Compression | Notes |
|----------|-------|-------------|-------|
| QK-reuse (thresh=0.75, rank=16) | 60.7% | 1.44× | Only 3/12 heads usable |
| Low-rank (rank=501, 95% energy) | 22.3% | 1.15× | Modest compression |
| Low-rank (rank=256) | 47.5% | 2.25× | |
| INT8 per-channel | **1.28%** | **4.0×** | **Best single method** |
| INT4 group (64) | 10.9% | 8.0× | Good 8-bit budget option |
| FP8 | **2.65%** | **4.0×** | Excellent accuracy/compression |
| **Pipeline B: LR(256) → INT4 group** | **48.4%** | **18.0×** | Best 18× compression |
| **Pipeline C: LR(256) → FP8** | **47.5%** | **9.0×** | |
| **Pipeline E: INT8 → LR(256)** | **47.5%** | **9.0×** | |

**→ Quantization dominates practical compression.** Low-rank methods add little compression beyond quantization for the same error budget. QK-reuse is too selective (only layer 0, few heads) to be broadly useful.

---

## Overall Conclusions

1. **Transformer weights are highly structured** — significantly lower effective rank and higher condition number than Gaussian matrices of same size
2. **Structure is not uniform across layers** — deeper layers are higher rank, better conditioned, less Q/K aligned
3. **Per-channel INT8 quantization is the practical winner** — consistently ~1.3% error at 4× compression across all layers and matrix types
4. **FP8 is a close second** — slightly more error (2.65%) but hardware-friendly
5. **Group INT4 (~11% error at 8×)** is viable for aggressive compression
6. **Low-rank + quantization pipelines** don't meaningfully outperform quantization alone at equivalent error
7. **Q/K weight reuse is a layer-0 phenomenon** — doesn't generalize across depth

---

## Running Experiments

```bash
# Activate venv (Windows)
.\.venv\Scripts\activate

# Run individual experiments
python experiments/E000_weight_analysis/run.py
python experiments/E006_multi_layer/run.py
python experiments/E007_quantization/run.py
python experiments/E008_combined_compression/run.py
```

Results are saved to `results/E###/summary.json` and plots to `results/E###/*.png`.

---

## Project Structure

```
genweight/
├── baselines.py          # Gaussian SVD baseline
├── block_pca.py          # Block-wise PCA
├── correlation.py        # Spatial correlation analysis
├── dictionary.py         # K-means codebook compression
├── generators/           # Neural weight generators
│   ├── coordinate.py     # Coordinate MLP generator
│   └── block.py          # Block latent generator
├── loader.py             # HuggingFace model loader
├── low_rank.py           # Low-rank SVD analysis
├── multi_layer.py        # Cross-layer analysis
├── quantization.py       # Quantization suite (INT4/8, FP4/8, group)
├── qkv.py                # Q/K/V block & head analysis
├── similarity.py         # Block cosine similarity
├── statistics.py         # Descriptive statistics
├── svd.py                # SVD spectrum analysis
└── visualization.py      # Histograms, heatmaps, spectra plots

experiments/
├── E000_weight_analysis/        # Core single-layer analysis
├── E001_coordinate_generator/   # Coordinate MLP generator
├── E002_block_dictionary/       # K-means codebook baseline
├── E003_block_generator/        # Block latent generator
├── E004_block_pca/              # Block PCA baseline
├── E005_qk_residual_compression/# Q→K transform + low-rank residual
├── E006_multi_layer/            # All 12 attention layers
├── E007_quantization/           # Quantization sensitivity
└── E008_combined_compression/   # Combined pipelines
```

---

## Requirements

- Python 3.12+
- PyTorch 2.13+
- Transformers 5.14+
- NumPy, SciPy, scikit-learn, Matplotlib, Pandas

Install:
```bash
pip install -e .
```

---

## License

MIT