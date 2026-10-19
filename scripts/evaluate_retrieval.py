
"""Evaluate retrieval quality using expected source URLs."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parents[1]
TOP_K = 4

SOURCE_EMBEDDINGS = (
    "https://docs.langchain.com/oss/python/integrations/text_embedding"
)
SOURCE_VECTORSTORES = (
    "https://docs.langchain.com/oss/python/integrations/vectorstores"
)
SOURCE_AGENTIC_BLOG = (
    "https://blog.langchain.com/agentic-rag-with-langgraph/"
)
SOURCE_AGENTIC_DOCS = (
    "https://docs.langchain.com/oss/python/langgraph/agentic-rag"
)
SOURCE_OVERVIEW = (
    "https://docs.langchain.com/oss/python/langchain/overview"
)
SOURCE_FUTURESMART = (
    "https://futuresmart.ai/blog/"
    "langgraph-rag-agent-tutorial-basics-to-advanced-multi-agent-ai-chatbot"
)


TEST_CASES = [
    {
        "question": "How do embeddings work?",
        "expected_sources": [SOURCE_EMBEDDINGS],
    },
    {
        "question": "What is a vector store?",
        "expected_sources": [SOURCE_VECTORSTORES],
    },
    {
        "question": "How does an agentic RAG workflow work?",
        "expected_sources": [SOURCE_AGENTIC_BLOG, SOURCE_AGENTIC_DOCS],
    },
    {
        "question": "What happens when retrieved documents are irrelevant?",
        "expected_sources": [SOURCE_AGENTIC_BLOG],
    },
    {
        "question": (
            "How does LangChain create_agent use middleware for "
            "guardrails, retries, routing, and tool policies?"
        ),
        "expected_sources": [SOURCE_OVERVIEW],
    },
    {
    "question": (
        "How does a LangGraph RAG agent combine tools, "
        "memory, routing, and document retrieval?"
    ),
    "expected_sources": [
        SOURCE_AGENTIC_BLOG,
        SOURCE_AGENTIC_DOCS,
        SOURCE_FUTURESMART,
    ],
},
{
    "question": (
        "How do the chatbot node and tool node cooperate "
        "to decide whether a LangGraph agent should call a tool?"
    ),
    "expected_sources": [SOURCE_FUTURESMART],
},
]


def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma(
        persist_directory=str(BASE_DIR / "chroma_db"),
        embedding_function=embeddings,
        collection_name="technical_docs",
    )

    print("Indexed chunks:", vector_db._collection.count())
    print(f"Evaluation: {len(TEST_CASES)} questions, top-{TOP_K} retrieval")
    print("Metric: expected source found in the top results\n")

    hits = 0
    reciprocal_ranks = []

    for index, test_case in enumerate(TEST_CASES, start=1):
        question = test_case["question"]
        expected_sources = set(test_case["expected_sources"])

        results = vector_db.similarity_search_with_score(
            question,
            k=TOP_K,
        )

        matching_rank = None

        print("=" * 75)
        print(f"TEST {index}: {question}")

        for rank, (doc, distance) in enumerate(results, start=1):
            source = doc.metadata.get("source", "Unknown")

            if source in expected_sources and matching_rank is None:
                matching_rank = rank

            print(f"\nRank: {rank} | Distance: {distance:.4f}")
            print(f"Source: {source}")
            print(
                doc.page_content[:250].replace("\n", " ")
            )

        if matching_rank is not None:
            hits += 1
            reciprocal_ranks.append(1 / matching_rank)
            print(
                f"\nPASS: An expected source appeared at rank "
                f"{matching_rank}."
            )
        else:
            reciprocal_ranks.append(0.0)
            print(
                f"\nMISS: No expected source appeared in the "
                f"top {TOP_K} results."
            )

    hit_rate = hits / len(TEST_CASES)
    mean_reciprocal_rank = sum(reciprocal_ranks) / len(TEST_CASES)

    print("\n" + "=" * 75)
    print("EVALUATION SUMMARY")
    print(f"Questions evaluated: {len(TEST_CASES)}")
    print(f"Expected-source hits: {hits}/{len(TEST_CASES)}")
    print(f"Hit rate @ {TOP_K}: {hit_rate:.1%}")
    print(f"Mean reciprocal rank (MRR): {mean_reciprocal_rank:.3f}")
    print(
        "\nNote: These metrics use manually selected expected source URLs. "
        "They measure source retrieval, not answer correctness or "
        "citation support."
    )


if __name__ == "__main__":
    main()
