
"""Create and persist a fresh Chroma vector store."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parents[2]
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "technical_docs"


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Remove the previous collection to prevent duplicate indexing.
    if CHROMA_DIR.exists():
        existing_db = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
        existing_db.delete_collection()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME,
    )

    print(f"ChromaDB saved to: {CHROMA_DIR}")
    print(f"Indexed chunks: {vector_store._collection.count()}")

    return vector_store
