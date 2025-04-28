import os
import pickle
from chromadb import PersistentClient  # new client API

# Path to your serialized embeddings
EMB_FILE = os.path.join(os.path.dirname(__file__), "embeddings.pkl")
# On-disk store directory
STORE_DIR = os.path.join(os.path.dirname(__file__), "..", ".chromastore")

def run():
    # Initialize a persistent Chroma client
    client = PersistentClient(path=STORE_DIR)

    # Create (or get) the collection for supply-chain docs
    collection = client.get_or_create_collection(name="supply_chain")

    # Load embeddings and document texts
    with open(EMB_FILE, "rb") as f:
        data = pickle.load(f)

    # Add each chunk into the collection
    for doc_id, info in data.items():
        collection.add(
            ids=[doc_id],
            documents=[info["text"]],
            embeddings=[info["embedding"].tolist()]
        )

    print(f"Indexed embeddings into ChromaDB at {STORE_DIR}")
