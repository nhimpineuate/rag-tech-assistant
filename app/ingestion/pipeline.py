"""Document ingestion pipeline."""

from pathlib import Path

from app.ingestion.loader import load_documents
from app.ingestion.splitter import split_documents
from app.ingestion.embeddings import create_vector_store

BASE_DIR = Path(__file__).resolve().parents[2]
URLS_FILE = BASE_DIR / "data" / "urls.txt"
DOCUMENTS_DIR = BASE_DIR / "documents"


def run_ingestion():

    documents = []

    if URLS_FILE.exists():
        web_documents = load_documents(str(URLS_FILE))
        documents.extend(web_documents)
    else:
        print(f"URLs file not found: {URLS_FILE}")

    from langchain_community.document_loaders import TextLoader

    for file_path in DOCUMENTS_DIR.glob("*.md"):
        print(f"Loading local file: {file_path.name}")
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents.extend(loader.load())

    if not documents:
        raise ValueError(
            "No documents loaded. Check data/urls.txt and documents/."
        )

    chunks = split_documents(documents)

    if not chunks:
        raise ValueError("No chunks were created.")

    create_vector_store(chunks)

    print(f"\nIngestion complete: {len(documents)} documents, "
          f"{len(chunks)} chunks.")