# Step 1: encode one dataset with one model and save everything to emb/<model>_<task>/.
# Usage: python step1_embed.py gemma FiQA2018
import json
import os
import sys

import mteb
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

MODELS = {"gemma": "google/embeddinggemma-300m", "qwen": "Qwen/Qwen3-Embedding-0.6B"}
# Task instruction for Qwen3 queries (same wording mteb uses for these tasks).
INSTRUCT = {
    "FiQA2018": "Given a financial question, retrieve user replies that best answer the question",
    "SciFact": "Given a scientific claim, retrieve documents that support or refute the claim",
}

name, task_name = sys.argv[1], sys.argv[2]
out = f"emb/{name}_{task_name}"

task = mteb.get_task(task_name)
task.load_data()
split = task.dataset["default"]["test"]
corpus, queries, qrels = split["corpus"], split["queries"], split["relevant_docs"]

# Same text join as MTEB and SpecTemp: "title text".
titles = corpus["title"] if "title" in corpus.column_names else [""] * len(corpus)
docs = [f"{t} {x}".strip() if t else x for t, x in zip(titles, corpus["text"])]

if name == "gemma":
    # encode_query / encode_document add Gemma's own retrieval prompts; output is mean-pooled and normalised.
    model = SentenceTransformer(MODELS[name])
    Q = model.encode_query(queries["text"], batch_size=64, show_progress_bar=True)
    D = model.encode_document(docs, batch_size=16, show_progress_bar=True)
else:
    # Qwen3: last-token pooling, official "Instruct: ...\nQuery:" prompt on queries only, documents plain.
    model = SentenceTransformer(MODELS[name], model_kwargs={"torch_dtype": torch.bfloat16})
    model.max_seq_length = 2048
    prompt = f"Instruct: {INSTRUCT[task_name]}\nQuery:"
    Q = model.encode(queries["text"], prompt=prompt, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    D = model.encode(docs, batch_size=8, normalize_embeddings=True, show_progress_bar=True)

Q, D = np.asarray(Q, dtype=np.float32), np.asarray(D, dtype=np.float32)
os.makedirs(out, exist_ok=True)
np.save(f"{out}/Q.npy", Q)
np.save(f"{out}/D.npy", D)
meta = {
    "qids": list(queries["id"]),
    "dids": list(corpus["id"]),
    "qrels": {q: dict(v) for q, v in qrels.items()},
}
with open(f"{out}/meta.json", "w") as f:
    json.dump(meta, f)

print(out, "Q", Q.shape, "D", D.shape, "norm of first doc", np.linalg.norm(D[0]))
