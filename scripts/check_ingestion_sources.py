
"""Check URL loading and document chunking without modifying ChromaDB."""

from collections import Counter
from pathlib import Path

from app.ingestion.loader import load_documents
from app.ingestion.splitter import split_documents


BASE_DIR = Path(__file__).resolve().parents[1]
URLS_FILE = BASE_DIR / "data" / "urls.txt"


def main():
    documents = load_documents(str(URLS_FILE))

    if not documents:
        raise SystemExit(
            "No web documents loaded. ChromaDB was not modified."
        )

    chunks = split_documents(documents)

    print("\n========== INGESTION CHECK ==========")
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    chunk_counts = Counter(
        chunk.metadata.get("source", "Unknown")
        for chunk in chunks
    )

    print("\nChunks per source:")
    for source, count in chunk_counts.items():
        print(f"- {source}: {count}")

    print("\nCheck complete. ChromaDB was not modified.")


if __name__ == "__main__":
    main()
