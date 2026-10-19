
"""Split documents into smaller, section-aware chunks."""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=[
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
        keep_separator="start",
    )

    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks
