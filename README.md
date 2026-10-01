# CS613 Assignment 2

Post-hoc embedding post-processing for retrieval: none, R1, R2 (Ren et al., 2025), SpecTemp (Li et al., 2026), PCA, whitening, and our R2 + SpecTemp variant. Models: EmbeddingGemma-300m, Qwen3-Embedding-0.6B. Datasets: FiQA2018, SciFact (MTEB). Metrics: nDCG@10 (main), Recall@100, average cosine.

## Setup

```bash
pip install -r requirements.txt
hf auth login   # EmbeddingGemma is gated
```

## Run

```bash
python step1_embed.py gemma FiQA2018   # encode once per model and dataset, saves emb/<model>_<task>/
python step1_embed.py gemma SciFact
python step1_embed.py qwen FiQA2018
python step1_embed.py qwen SciFact
python scoring.py emb/gemma_FiQA2018   # baseline check
python step3_ladder.py                 # all methods, writes results/results.csv
python count_params.py                 # parameter counts
```
