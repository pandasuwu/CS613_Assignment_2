# Representation Learning via Embedding Post-Processing

Research codebase for training-free, parameter-free geometric post-processing of neural text representations across dedicated embedding models and base language models on MTEB v2.

## Evaluated Models

Dedicated Embedding Models:
- google/embeddinggemma-300m (308M parameters, d=768)
- Qwen/Qwen3-Embedding-0.6B (600M parameters, d=1024)

Base Language Models (evaluated under mean and last-token pooling):
- google/gemma-3-1b-pt (1.0B parameters, d=1152)
- Qwen/Qwen3.5-0.8B-Base (0.8B parameters, d=1024)

## Benchmarks (MTEB v2 English)

Information Retrieval:
- Development core: FiQA2018, ArguAna, SCIDOCS, TRECCOVID
- Scale-up suite: FEVERHardNegatives, ClimateFEVERHardNegatives, HotpotQAHardNegatives, Touche2020Retrieval.v3, CQADupstackGamingRetrieval, CQADupstackUnixRetrieval

Semantic Textual Similarity (STS):
- STSBenchmark, SICK-R, STS22.v2

## Post-Processing Methods

Full-Dimension Transforms (d to d):
- baseline: Raw unit-normalized representations
- standardization: Coordinate-wise z-score scaling (Timkey and van Schijndel, 2021)
- r1: Direct empirical mean subtraction (Ren et al., 2025)
- r2: Mean-subspace projection deflation (Ren et al., 2025)
- soft_zca: Regularized ZCA whitening (Diera et al., 2024)
- abtt_1, abtt_2, abtt_3: Centered PCA deflation of top 1, 2, or 3 components (Mu and Viswanath, 2018)
- rand: Random unit direction deflation negative control (Ren et al., 2025)
- mc: Unnormalized mean centering negative control (Ren et al., 2025)

Compression Transforms (d to k for k in 512, 256, 128, 64):
- prefix: Coordinate prefix slicing (Matryoshka baseline)
- random_truncation: Uniform random coordinate sampling negative control
- pca: Top-k principal component projection (Li et al., 2026)
- whitening: Truncated Whitening-k variance equalization (Su et al., 2021)
- spectemp: Adaptive SNR spectral tempering via Kneedle knee detection (Li et al., 2026)

## Metrics

- Retrieval: nDCG@10 (primary), Recall@100, MRR@10
- Similarity: Spearman rank correlation rho (primary), Pearson correlation r
- Intrinsic Geometry: centroid_norm, average_cosine, mev_top1, nid_spectral_entropy, isoscore

## Setup

Requires Python 3.14 and uv:

```bash
uv sync
```

All models, tokenizers, and datasets are isolated in project-local cache/ directory.

## Execution Pipeline

The execution operates in three decoupled, resumable stages.

### Stage 1: Encode and Cache Raw Representations

Extracts representations once per model, task, and pooling mode into cache/embeddings/:

```bash
# Run all models and core tasks
uv run encode.py

# Run a single combination
uv run encode.py --model google/embeddinggemma-300m --task FiQA2018
uv run encode.py --model Qwen/Qwen3.5-0.8B-Base --task FiQA2018 --pooling mean_pooling
```

### Stage 2: Post-Processing Transformations

Applies all full and compression transformations to raw cached tensors:

```bash
# Transform all cached combinations
uv run transform.py

# Transform a single combination
uv run transform.py --model google/embeddinggemma-300m --task FiQA2018
```

### Stage 3: Evaluation and Metric Generation

Evaluates transformed representations and writes atomic CSV leaves into results/:

```bash
# Evaluate information retrieval
uv run eval_retrieval.py

# Evaluate semantic textual similarity
uv run eval_similarity.py

# Evaluate intrinsic space geometry diagnostics
uv run eval_geometry.py
```

Pass `--overwrite` to any script to recompute existing cached artifacts or results.

## Code Quality

```bash
uv run ruff check .
uv run ty check
```
