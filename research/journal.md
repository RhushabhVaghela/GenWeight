# GenWeight Research Journal

This journal records the scientific story of GenWeight. Each entry is intentionally short: question, method, result, interpretation, and next step.

## Checkpoint: Current Research State

Date: 2026-08-04

Scope: GPT-2 small, dense weight geometry and compression.

Core question:

> Can transformer weights be explained or compressed by a smaller mathematical representation than explicit dense matrices?

Current answer:

> GPT-2 weights are structured, but the practical compression path is currently quantization, not the simple generative representations tested so far.

## E000: Single Matrix Geometry

Question: Is GPT-2 layer 0 `c_attn` random or structured?

Method: Descriptive statistics, SVD, Gaussian baseline, spatial correlation, block similarity, Q/K/V block analysis.

Results:

- Effective rank: 394 vs 650 for matched Gaussian.
- Energy rank 90%: 399 vs 559 for matched Gaussian.
- Local spatial correlations were near zero.
- Strong block similarity appeared mainly between aligned Q/K regions.

Interpretation: The matrix is not random, but the structure is spectral and module-specific rather than local pixel-like smoothness.

Next step: Test whether the discovered structure generalizes beyond layer 0.

## E001: Coordinate Generator

Question: Can a small coordinate model predict weights from row and column indices?

Method: Train a coordinate-based neural generator on `h.0.attn.c_attn.weight`.

Result: About 100% relative reconstruction error with about 5.8% parameter budget.

Interpretation: The weight matrix does not behave like a simple smooth coordinate field at this scale.

Next step: Try block-level structure instead of individual coordinate prediction.

## E002: Block Dictionary

Question: Can repeated block prototypes compress the matrix?

Method: K-means/codebook compression over 64x64 blocks.

Result: 86-98% relative error for tested codebook sizes.

Interpretation: Blocks have too much residual variation for a tiny prototype dictionary.

Next step: Use learned continuous latents instead of hard block IDs.

## E003: Latent Block Generator

Question: Can each block be represented by a small latent code plus shared decoder?

Method: Train latent block generator over 64x64 blocks.

Result: 74.8% relative error after 5000 steps at about 15.9% parameter budget.

Interpretation: The generator learns something, but it is still worse than PCA at similar storage.

Next step: Compare with explicit block PCA.

## E004: Block PCA

Question: How strong is a simple linear block baseline?

Method: PCA over flattened blocks.

Result: 70.6% error at 16.6% storage; 51.8% error at 33.0% storage.

Interpretation: Linear block structure beats the first nonlinear block generator, but still has high error.

Next step: Focus on the strongest observed structure: Q/K reuse.

## E005: Q/K Residual Compression

Question: Can K heads be generated from matching Q heads plus a residual?

Method: Select high-cosine Q/K heads, fit linear Q-to-K transforms, then encode low-rank residuals.

Results:

- Selected heads: 1, 5, 10.
- Rank 16 residual: 12.3% error, 94.6% parameter ratio, about 1.06x compression.
- Exact rank 64 reconstruction costs more than dense storage.

Interpretation: Q/K reuse is real but too narrow to be a broadly useful compressor.

Next step: Test whether layer 0 is special.

## E006: Multi-Layer Structure

Question: Does the layer-0 structure generalize across GPT-2 attention layers?

Method: Analyze all 12 attention `c_attn` matrices.

Result: Effective rank generally increases with depth; Q/K high-similarity heads disappear after layer 0.

Interpretation: A single Q/K reuse rule will not generalize across the model.

Next step: Shift from structural reuse to quantization sensitivity.

## E007-E008: Quantization and Combined Pipelines

Question: Which basic compression methods are actually competitive?

Method: Quantize all major 2D matrices and test low-rank/quantization combinations.

Result: INT8 per-channel and FP8 were much stronger than low-rank, Q/K reuse, or combined low-rank pipelines.

Interpretation: Quantization dominates practical compression so far.

Next step: Explore smarter quantization families and wider benchmarks.

## E009-E016: Smart Quantization and Broad Benchmark

Question: Do GPTQ-like, AWQ-like, AutoRound, NF4, GGUF, and NVFP4 methods improve over simple quantization?

Method: Evaluate many quantization schemes across 48 GPT-2 2D matrices.

Result: FP8 and INT8 per-channel dominate weight reconstruction; NVFP4 and group INT4 are strongest among tested 4-bit families.

Interpretation: More elaborate schemes are not automatically better without strong calibration and careful implementation.

Next step: Add production-style per-channel baselines and behavior-level evaluation.

## E017: Per-Channel Quantization

Question: How do per-channel and group-wise quantizers compare across GPT-2?

Method: Compare INT8 per-channel, INT4 per-channel, group INT4, GGUF Q4_K, and NVFP4.

Results:

- INT8 per-channel: 1.01% average Frobenius error.
- NVFP4: 9.30% average error.
- INT4 group g=32: 9.99% average error.

Interpretation: INT8 per-channel is the strongest reconstruction baseline; 4-bit still has substantial error.

Next step: Test whether these reconstruction errors preserve model behavior.

## E019: End-to-End Perplexity

Question: Do quantized weights preserve GPT-2 next-token behavior?

Method: Quantize all major 2D weights and evaluate perplexity on a small local English corpus.

Results:

- FP32 baseline perplexity: 95.23.
- FP8 perplexity: 93.12.
- INT8 per-channel perplexity: 103.87.
- INT4 group g=32 perplexity: 1205.43.
- NVFP4 perplexity: 7515.54.

Interpretation: FP8 is currently the best behavior-preserving method. Uniform 4-bit quantization is not safe despite moderately acceptable Frobenius error.

Next step: Try mixed precision: keep sensitive layers at FP8/INT8 and use 4-bit only where safe.

## E018: True-Hessian GPTQ-Like Calibration

Question: Can real activation Hessian information improve 4-bit quantization?

Status: Implemented, pending validation.

Interpretation caution: The current implementation is GPTQ-like because it uses calibration Hessian information, but it should not yet be treated as full GPTQ unless error compensation and ordering are implemented carefully.

Next step: Run and compare against NVFP4, group INT4, and FP8 behavior-level baselines.

## E020: Mixed Precision

Question: Can sensitivity-aware bit allocation preserve behavior while recovering some 4-bit memory savings?

Status: Implemented, pending validation.

Hypothesis: Uniform 4-bit fails because some layers are behavior-critical. Mixed precision may work if sensitive modules remain FP8 or INT8.

Next step: Run E020, then validate the best candidates with the E019 perplexity framework.
