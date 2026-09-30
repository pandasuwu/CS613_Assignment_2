# Step 1: encode FiQA2018 once with EmbeddingGemma and save everything to emb/.
import json
import os

import mteb
import numpy as np
from sentence_transformers import SentenceTransformer

task = mteb.get_task("FiQA2018")
task.load_data()
split = task.dataset["default"]["test"]
corpus, queries, qrels = split["corpus"], split["queries"], split["relevant_docs"]

# Same text join as MTEB and SpecTemp: "title text".
titles = corpus["title"] if "title" in corpus.column_names else [""] * len(corpus)
docs = [f"{t} {x}".strip() if t else x for t, x in zip(titles, corpus["text"])]

# encode_query / encode_document add Gemma's own retrieval prompts.
# The output is already mean-pooled and L2-normalised by the model.
model = SentenceTransformer("google/embeddinggemma-300m")
Q = model.encode_query(queries["text"], batch_size=64, show_progress_bar=True)
D = model.encode_document(docs, batch_size=16, show_progress_bar=True)

os.makedirs("emb", exist_ok=True)
np.save("emb/Q.npy", Q)
np.save("emb/D.npy", D)
meta = {
    "qids": list(queries["id"]),
    "dids": list(corpus["id"]),
    "qrels": {q: dict(v) for q, v in qrels.items()},
}
with open("emb/meta.json", "w") as f:
    json.dump(meta, f)

print("Q", Q.shape, "D", D.shape, "norm of first doc", np.linalg.norm(D[0]))
