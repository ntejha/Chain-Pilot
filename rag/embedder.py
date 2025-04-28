import os, pickle
from sentence_transformers import SentenceTransformer

EMB_MODEL  = "all-MiniLM-L6-v2"
CHUNKS_DIR = os.path.join(os.path.dirname(__file__), "chunks")
EMB_OUT    = os.path.join(os.path.dirname(__file__), "embeddings.pkl")

def run():
    model = SentenceTransformer(EMB_MODEL)
    embeddings = {}
    for fn in os.listdir(CHUNKS_DIR):
        path = os.path.join(CHUNKS_DIR, fn)
        text = open(path, "r").read()
        emb = model.encode(text)
        embeddings[fn] = {"text": text, "embedding": emb}
    with open(EMB_OUT, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Saved embeddings to {EMB_OUT}")
