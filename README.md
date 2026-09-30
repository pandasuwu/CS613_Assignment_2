# CS613 Assignment 2

Compares none, R1, R2 and SpecTemp on EmbeddingGemma embeddings of FiQA2018 (nDCG@10).

## Setup

```bash
pip install -r requirements.txt
hf auth login   # EmbeddingGemma is gated
```

## Run

```bash
python step1_embed.py    # encode once, saves emb/
python scoring.py        # baseline, should be close to 0.4774
python step3_ladder.py   # all methods vs none
```
