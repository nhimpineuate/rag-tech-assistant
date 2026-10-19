"""Load main article content from URLs."""
from pathlib import Path
import trafilatura
from langchain_core.documents import Document
def load_documents(url_file: str):
    """Extract readable article text from each URL."""
    with open(url_file, "r", encoding="utf-8") as file:
        urls = [
            line.strip()
            for line in file
            if line.strip() and not line.strip().startswith("#")
        ]
    documents = []
    for url in urls:
        print(f"Loading: {url}")
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                print("  Failed to download page.")
                continue

            content = trafilatura.extract(
                downloaded,
                output_format="txt",
                include_comments=False,
                include_tables=True,
                favor_precision=True,
            )

            if not content or len(content.strip()) < 200:
                print("  Insufficient main content; skipped.")
                continue
            documents.append(
                Document(
                    page_content=content.strip(),
                    metadata={"source": url},
                )
            )
            print(f"  Extracted {len(content)} characters")
        except Exception as exc:
            print(f"  Failed to load page: {exc}")
    print(f"\nLoaded {len(documents)} web documents.")
    return documents
