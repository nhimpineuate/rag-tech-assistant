"""Inspect indexed Chroma chunks and their metadata."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory=str(BASE_DIR / "chroma_db"),
    embedding_function=embeddings,
    collection_name="technical_docs",
)

data = vector_db.get(include=["documents", "metadatas"])

print(f"Total chunks: {len(data['documents'])}")

for i, (document, metadata) in enumerate(
    zip(data["documents"], data["metadatas"]), start=1
):
    print(f"\n{'=' * 70}")
    print(f"Chunk {i}")
    print(f"Source: {metadata.get('source', 'Unknown')}")
    print(f"Characters: {len(document)}")
    print(document[:350].replace("\n", " "))
