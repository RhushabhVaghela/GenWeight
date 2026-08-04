# GenWeight Research Summary

Project: Transformer weight geometry and compression analysis

Model: GPT-2 small, 124M parameters

Current status: active research prototype through E020

## Executive Summary

GenWeight began as an investigation into whether transformer weights can be represented by a compact generative object rather than stored as dense matrices. The evidence so far is mixed in an interesting way:

- GPT-2 weights are clearly structured and differ strongly from matched Gaussian matrices.
- The simple generative representations tested so far do not compress weights competitively.
- Practical compression is currently dominated by quantization.
- End-to-end behavior is stricter than weight reconstruction error: uniform 4-bit methods damage perplexity badly, while FP8 preserves behavior best in the current test.

The strongest current result is E019: FP8 E4M3 produced 2.65% average Frobenius error and slightly improved measured perplexity on the small built-in test corpus, while 4-bit methods with 9-12% Frobenius error caused very large perplexity degradation.

## Current Best Evidence

### Behavioral Compression, E019

| Scheme | Avg Frobenius Error | Perplexity | Change vs FP32 |
|---|---:|---:|---:|
| FP32 baseline | - | 95.23 | - |
| FP8 E4M3 | 2.65% | 93.12 | -2.22% |
| INT8 per-channel | 1.01% | 103.87 | +9.07% |
| INT4 group, g=32 | 9.99% | 1205.43 | +1165.76% |
| INT4 group, g=64 | 11.29% | 1908.20 | +1903.70% |
| NVFP4 | 9.30% | 7515.54 | +7791.68% |
| GGUF Q4_K | 12.25% | 30198.05 | +31609.40% |
| INT4 per-channel | 17.11% | 586463.79 | +615715.17% |

Conclusion: FP8 is the current practical winner. INT8 per-channel reconstructs weights most accurately but caused a modest perplexity increase in this small evaluation. Uniform 4-bit quantization is not safe yet.

### Weight-Space Quantization, E017

| Scheme | Avg Error | Main Takeaway |
|---|---:|---|
| INT8 per-channel | 1.01% | Best reconstruction |
| NVFP4 | 9.30% | Best tested 4-bit weight error |
| INT4 group, g=32 | 9.99% | Strong group-wise baseline |
| INT4 group, g=64 | 11.29% | Slightly worse than g=32 |
| GGUF Q4_K | 12.25% | Sensitive on some matrices |
| INT4 per-channel | 17.11% | Too coarse |

Conclusion: Weight error alone made some 4-bit methods look promising, but E019 showed that behavior can still collapse. Future compression must include perplexity or task-level checks.

### Weight Geometry, E000 and E006

Layer 0 `h.0.attn.c_attn.weight` is more structured than a matched Gaussian baseline:

- Effective rank: 394 vs 650 Gaussian
- Condition number: 45.5 vs about 3.7 Gaussian
- Energy rank 90%: 399 vs 559 Gaussian
- Energy rank 95%: 501 vs 641 Gaussian

Across attention layers, the structure changes:

- Effective rank generally increases with depth.
- Condition number generally decreases with depth.
- Q/K reuse is strong only in layer 0 and only for a few heads.

Conclusion: structure exists, but it is not uniform enough for one simple global compression rule.

### Generative and Structural Compression, E001-E005

| Experiment | Method | Result |
|---|---|---|
| E001 | Coordinate MLP | About 100% error; simple coordinate field failed |
| E002 | Block dictionary | 86-98% error; dictionary too coarse |
| E003 | Latent block generator | 74.8% error at 15.9% params after 5000 steps |
| E004 | Block PCA | 70.6% error at 16.6% storage |
| E005 | Q/K residual reuse | Real layer-0 Q/K reuse, but only about 1.06x compression at rank 16 |

Conclusion: the original generative hypothesis remains scientifically interesting, but the naive implementations are not competitive.

## Experiment Ledger

| ID | Status | Question | Outcome |
|---|---|---|---|
| E000 | Complete | Is one GPT-2 matrix random or structured? | Structured: low effective rank and high anisotropy vs Gaussian |
| E001 | Complete | Can coordinates predict weights? | No useful reconstruction at tested size |
| E002 | Complete | Can block dictionaries compress weights? | Not competitive |
| E003 | Complete | Can a latent block generator learn weight blocks? | Learns slowly, still high error |
| E004 | Complete | Is block PCA a stronger baseline? | Yes, but still too lossy |
| E005 | Complete | Can K be generated from Q plus residual? | Only for 3 layer-0 heads, weak compression |
| E006 | Complete | Does layer-0 structure generalize? | No; deeper layers differ substantially |
| E007 | Complete | How sensitive are all 2D matrices to simple quantization? | INT8/FP8 strong; INT4 mixed |
| E008 | Complete | Do combined pipelines beat quantization? | No meaningful win over direct quantization |
| E009 | Complete | Do GPTQ/AWQ-style prototypes help? | Not yet; proxy methods underperform |
| E010 | Complete | Does activation calibration improve AWQ? | Explored, not yet a clear win |
| E016 | Complete | What happens across 15 schemes and 48 matrices? | FP8 and INT8 per-channel dominate weight-space error |
| E017 | Complete | How do per-channel and group-wise schemes compare? | INT8 per-channel best; NVFP4 best 4-bit by error |
| E018 | Implemented | Can true calibration Hessians improve GPTQ-like quantization? | Pending validation |
| E019 | Complete | Do quantized weights preserve perplexity? | FP8 preserves behavior; uniform 4-bit fails |
| E020 | Implemented | Can mixed precision recover 4-bit savings safely? | Pending validation |

## Current Direction

The next scientifically justified step is E020 mixed precision.

Reason:

1. E017 shows some 4-bit methods are numerically decent.
2. E019 shows uniform 4-bit quantization destroys behavior.
3. Therefore the next hypothesis is sensitivity-aware allocation: keep behavior-critical layers at FP8 or INT8 and use 4-bit only where the model tolerates it.

After E020, the next credibility upgrade is a standard perplexity dataset such as WikiText-2, because the current E019 corpus is intentionally small and local.

## Research Rules Going Forward

1. Every new algorithmic hypothesis should be motivated by an experiment.
2. Every compression claim must report both weight error and behavior-level quality.
3. Every experiment should produce code, numeric output, a written interpretation, and a commit.
4. Random or matched baselines should be used whenever a claim says GPT-2 is special.
5. Results should be documented in `research/journal.md` before being generalized into README conclusions.
