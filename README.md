# GenWeight

Research framework for studying GPT-2 weight geometry, learned weight structure, and practical compression methods.

GenWeight started with a simple research question:

> Can transformer weights be represented by a smaller mathematical object instead of storing every dense parameter directly?

The project has now tested that idea from several angles: low-rank structure, block reuse, coordinate generators, dictionary compression, Q/K reuse, smart quantization, and end-to-end perplexity.

## Current Status

Status: active research prototype

Model studied so far: GPT-2 small, 124M parameters

Main conclusion so far: GPT-2 weights are structured, but practical compression is dominated by quantization, especially FP8 and INT8 per-channel.

The most important update is Experiment E019: Frobenius error alone is not enough. Several 4-bit methods looked numerically acceptable at the weight level but destroyed perplexity. FP8 preserved behavior best in the current CPU-scale evaluation.

## Headline Results

### End-to-End Perplexity, E019

This is currently the most important experiment because it measures model behavior, not just weight reconstruction error.

| Scheme | Avg Frobenius Error | Loss | Perplexity | Perplexity Change |
|---|---:|---:|---:|---:|
| Baseline FP32 | - | 4.5563 | 95.23 | - |
| FP8 E4M3 | 2.65% | 4.5338 | 93.12 | -2.22% |
| INT8 per-channel | 1.01% | 4.6432 | 103.87 | +9.07% |
| INT4 group, g=32 | 9.99% | 7.0946 | 1205.43 | +1165.76% |
| INT4 group, g=64 | 11.29% | 7.5539 | 1908.20 | +1903.70% |
| NVFP4 | 9.30% | 8.9247 | 7515.54 | +7791.68% |
| GGUF Q4_K | 12.25% | 10.3155 | 30198.05 | +31609.40% |
| INT4 per-channel | 17.11% | 13.2819 | 586463.79 | +615715.17% |

Interpretation:

- FP8 is currently the best practical scheme tested: modest weight error and no perplexity degradation on the current test corpus.
- INT8 per-channel has the lowest Frobenius error, but still increases perplexity by about 9% in this small evaluation.
- Current simulated 4-bit methods are not behaviorally safe for GPT-2 when applied uniformly to all major 2D weights.
- Weight-space error does not reliably predict model quality once errors reach the 4-bit range.

Important caveat: E019 uses a small built-in English sentence set, not WikiText or a full benchmark suite. It is useful for direction, but not a final deployment claim.

### Quantization Error, E017

E017 added production-style per-channel quantization and compared it to group-wise 4-bit formats across GPT-2 weight matrices.

| Scheme | Avg Error | Min Error | Max Error | Notes |
|---|---:|---:|---:|---|
| INT8 per-channel | 1.01% | 0.77% | 1.93% | Best weight reconstruction |
| NVFP4 | 9.30% | 5.98% | 9.57% | Best 4-bit weight error among tested schemes |
| INT4 group, g=32 | 9.99% | 8.18% | 11.23% | Strong 4-bit baseline by Frobenius error |
| INT4 group, g=64 | 11.29% | 9.71% | 13.49% | Slightly worse than g=32 |
| GGUF Q4_K | 12.25% | 10.94% | 25.94% | Sensitive on positional embeddings |
| INT4 per-channel | 17.11% | 13.59% | 26.79% | Too coarse per output channel |

Interpretation:

- Per-channel INT8 is extremely strong numerically.
- Fine-grained 4-bit grouping beats per-channel 4-bit.
- 4-bit Frobenius error around 9-12% is still too high for uniform end-to-end GPT-2 behavior in E019.

### Structure vs Randomness, E000 and E006

Layer 0 attention input projection, `h.0.attn.c_attn.weight`, is significantly more structured than a matched Gaussian matrix.

| Property | GPT-2 Layer 0 | Matched Gaussian | Meaning |
|---|---:|---:|---|
| Effective rank | 394 | 650 | GPT-2 has concentrated spectral energy |
| Condition number | 45.5 | about 3.7 | GPT-2 is much more anisotropic |
| Energy rank 90% | 399 | 559 | Lower-dimensional structure exists |
| Energy rank 95% | 501 | 641 | Structure remains at higher energy threshold |

Across all 12 GPT-2 attention layers, structure changes with depth:

- Effective rank generally increases with depth, roughly 394 to 577.
- Condition number generally decreases, roughly 45 to 9.
- Strong Q/K reuse appears mostly in layer 0.
- Later layers are less compressible by simple low-rank or Q/K reuse assumptions.

### Generative Compression Attempts, E001-E004

These experiments tested the original GenWeight idea: learn a compact representation that regenerates weights.

| Experiment | Method | Storage / Params | Error | Result |
|---|---:|---:|---:|---|
| E001 | Coordinate MLP, row/column to weight | 5.8% params | about 100% | Failed to learn useful field |
| E002 | Block dictionary / K-means | 0.9-7.4% storage | 86-98% | Too much residual variation |
| E003 | Latent block generator | 15.9% params | 74.8% after 5000 steps | Learns slowly, worse than PCA |
| E004 | Block PCA | 16.6% storage | 70.6% | Stronger than simple generator |

Interpretation:

- Raw GPT-2 weights do not behave like a smooth coordinate field.
- Simple learned generators are not competitive yet.
- The original hypothesis is not dead, but the naive versions are falsified.

### Q/K Reuse, E005

Layer 0 has a real but narrow Q/K reuse pattern.

| Residual Rank | Error | Parameter Ratio | Compression |
|---:|---:|---:|---:|
| 0 | 18.25% | 92.36% | 1.08x |
| 16 | 12.33% | 94.62% | 1.06x |
| 32 | 8.96% | 96.88% | 1.03x |
| 64 | 0.00% | 101.39% | 0.99x |

Only Q/K heads 1, 5, and 10 were selected by the similarity threshold. This is scientifically interesting but not a broadly useful compression method because it barely compresses and does not generalize across layers.

### Combined Pipelines, E008

Low-rank plus quantization did not beat simple quantization in the useful error range.

| Pipeline | Error | Compression | Conclusion |
|---|---:|---:|---|
| INT8 per-channel | 1.28% | 4.0x | Best clean single method for layer 0 |
| FP8 | 2.65% | 4.0x | Excellent and behaviorally strong in E019 |
| INT4 group, g=64 | 10.9% | about 8.0x ideal | Good weight error but poor E019 perplexity |
| Low-rank rank 256 | 47.5% | 2.25x | Too much error |
| Low-rank rank 256 then INT4 group | 48.4% | 18.0x ideal | Compression high, quality too poor |

Interpretation: low-rank approximation captures structure, but not enough useful structure to compete with direct quantization.

## Experiment Catalog

| ID | Status | Focus | Main Result |
|---|---|---|---|
| E000 | Complete | Single-layer weight geometry | Layer 0 `c_attn` is highly non-Gaussian in spectral structure |
| E001 | Complete | Coordinate generator | Failed: about 100% reconstruction error |
| E002 | Complete | Block dictionary baseline | Failed broadly: 86-98% error |
| E003 | Complete | Latent block generator | Partial learning, 74.8% error at 15.9% params |
| E004 | Complete | Block PCA baseline | Better than generator, still high error |
| E005 | Complete | Q/K residual compression | Real but narrow Q/K reuse in layer 0 only |
| E006 | Complete | Multi-layer attention analysis | Layer 0 is special; structure changes by depth |
| E007 | Complete | Basic quantization | Per-channel INT8 and FP8 are strongest |
| E008 | Complete | Combined compression pipelines | Quantization dominates low-rank hybrids |
| E009 | Complete | Smart quantization prototypes | Proxy GPTQ/AWQ underperformed simple group quantization |
| E010 | Complete | AWQ calibration | Activation scaling explored, not yet better than simple baselines |
| E011-E015 | Covered in E009/E016 | SmoothQuant, AutoRound, NF4, GGUF, NVFP4 | NVFP4 and GGUF-style methods strongest among 4-bit family by weight error |
| E016 | Complete | Comprehensive quantization benchmark | 15 schemes across 48 matrices, 720+ configurations |
| E017 | Complete | Per-channel quantization benchmark | INT8 per-channel about 1.01% avg error; NVFP4 about 9.30% |
| E018 | Implemented, pending validation | True-Hessian GPTQ calibration | Ready to run; needs careful interpretation as GPTQ-like, not full GPTQ |
| E019 | Complete | End-to-end perplexity | FP8 preserved behavior best; uniform 4-bit methods collapsed perplexity |
| E020 | Implemented, pending validation | Mixed precision | Next experiment: allocate FP8/INT8/4-bit by sensitivity |

## Current Scientific Conclusions

1. GPT-2 weights are structured, not random.
2. The structure is heterogeneous: it varies strongly across layers and modules.
3. Simple coordinate, dictionary, and block-generator representations are not competitive yet.
4. Q/K reuse exists, but mostly in layer 0 and only for a few heads.
5. Low-rank methods reveal structure but are poor practical compressors at acceptable error.
6. FP8 is currently the best behavior-preserving compression method tested.
7. INT8 per-channel is the best weight-reconstruction method tested, but caused a small perplexity increase in E019.
8. Uniform 4-bit quantization is not currently safe for GPT-2 behavior, despite 9-12% average Frobenius error.
9. The next promising direction is mixed precision: protect sensitive layers with FP8/INT8 and use 4-bit only where behavior tolerates it.

## What We Are Doing Next

The next priority is E020 mixed-precision quantization.

Why:

- E019 showed uniform 4-bit quantization damages model behavior badly.
- E017 showed some 4-bit schemes have reasonable weight error.
- Therefore the likely path is not uniform 4-bit; it is sensitivity-aware bit allocation.

The main question for E020 is:

> Can we keep most of the memory savings of 4-bit quantization while protecting behavior-critical layers with FP8 or INT8?

After E020, the next serious upgrade is to evaluate perplexity on a standard dataset such as WikiText-2 instead of the current built-in sentence set.

## Running Experiments

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

Install the project in editable mode:

```bash
uv pip install -e .
```

Run core experiments:

```bash
python experiments\E000_weight_analysis\run.py
python experiments\E006_multi_layer\run.py
python experiments\E016_quantization_benchmark\run.py
python experiments\E017_per_channel_quantization\run.py
python experiments\E019_perplexity\run.py
python experiments\E020_mixed_precision\run.py
```

Experiment outputs are written under `results/`, which is intentionally ignored by Git. Key numbers are summarized in this README, `RESEARCH_SUMMARY.md`, and `research/journal.md`.

## Project Structure

```text
genweight/
  baselines.py          Gaussian SVD baseline
  block_pca.py          Block-wise PCA compression
  correlation.py        Spatial neighbor correlation
  dictionary.py         K-means/block dictionary compression
  loader.py             Hugging Face model loading utilities
  low_rank.py           Low-rank SVD reconstruction baselines
  multi_layer.py        Cross-layer GPT-2 analysis
  qkv.py                Q/K/V block and head analysis
  quantization.py       INT, FP, GPTQ-like, AWQ-like, NF4, GGUF, NVFP4 utilities
  similarity.py         Block cosine similarity
  statistics.py         Descriptive weight statistics
  svd.py                SVD spectrum analysis
  visualization.py      Histograms, heatmaps, spectrum plots
  generators/
    coordinate.py       Coordinate-based weight generator
    block.py            Latent block generator

experiments/
  E000_weight_analysis/
  E001_coordinate_generator/
  E002_block_dictionary/
  E003_block_generator/
  E004_block_pca/
  E005_qk_residual_compression/
  E006_multi_layer/
  E007_quantization/
  E008_combined_compression/
  E009_gptq_quantization/
  E009_smart_quantization/
  E010_awq_calibration/
  E016_quantization_benchmark/
  E017_per_channel_quantization/
  E018_real_gptq/
  E019_perplexity/
  E020_mixed_precision/
```

## Requirements

- Python 3.12+
- PyTorch
- Transformers
- NumPy
- SciPy
- scikit-learn
- Matplotlib
- Pandas
- tqdm

Install dependencies:

```bash
uv pip install -r requirements.txt
uv pip install -e .
```

## Notes

- Results are currently CPU-friendly and GPT-2-small focused.
- Compression ratios in early exploratory scripts are idealized unless otherwise noted; future work should include scale/metadata overhead precisely.
- Full claims about model quality should be based on standard benchmark datasets, not only the small E019 sentence set.
- New hypotheses should be added to `research/journal.md` only after an experiment motivates them.

## License

MIT
