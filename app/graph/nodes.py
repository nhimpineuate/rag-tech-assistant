import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.llm import llm

BASE_DIR = Path(__file__).resolve().parents[2]
CHROMA_DIR = BASE_DIR / "chroma_db"

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load Chroma database
vector_db = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings,
    collection_name="technical_docs",
)


def analyze_query(state):
    print(f"\nQuestion: {state['question']}")
    return state


def retrieve_documents(state):
    question = state.get("rewritten_question") or state["question"]

    start = time.perf_counter()

    docs = vector_db.similarity_search(question, k=8)

    elapsed = time.perf_counter() - start

    state["documents"] = docs

    print(f"Retrieved {len(docs)} documents")
    print(f"Retrieval time: {elapsed:.3f} seconds")

    return state



def grade_documents(state):
    question = state["question"]
    documents = state.get("documents", [])
    filtered_documents = []

    for i, doc in enumerate(documents, start=1):
        prompt = f"""
    You are a relevance grader for a technical documentation assistant.

Question:
{question}

Retrieved document:
{doc.page_content[:2000]}

Your task is to decide whether this document contains information
that can help answer the question.

Important rules:
- A document does NOT need to provide a complete definition.
- Partial explanations, useful examples, mechanisms, and workflow
    descriptions can be relevant.
- Judge relevance to the question, not whether the document uses
    exactly the same words as the question.
- For questions about Agentic RAG, relevant evidence can include
    an agent deciding when to retrieve, tool use, routing, document
    grading, query rewriting, retrying retrieval, and self-reflection.
- Answer NO only when the document provides no useful information
    for answering the question.

Reply with exactly one word: YES or NO.
"""

        try:
            response = llm.invoke(prompt)
            decision = response.content.strip().upper()

            # Diagnostic: show the model's actual judgment.
            print(f"\nDocument {i} | Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Grader response: {response.content!r}")

            if decision.startswith("YES"):
                filtered_documents.append(doc)
                print(f"Document {i}: RELEVANT")
            elif decision.startswith("NO"):
                print(f"Document {i}: NOT RELEVANT")
            else:
                print(f"Document {i}: UNCLEAR RESPONSE")

        except Exception as e:
            print(f"Document {i}: Grading failed: {e}")

    state["filtered_documents"] = filtered_documents

    print(
        f"\nDocuments retrieved: {len(documents)}"
        f"\nDocuments passed to generation: "
        f"{len(filtered_documents)}"
    )

    return state


def decide_after_grading(state):
    """Choose generation or query rewriting based on relevance."""
    if state.get("filtered_documents"):
        return "generate"

    if state.get("retries", 0) < 2:
        return "rewrite"

    return "generate"


def rewrite_query(state):
    """Rewrite the question to improve retrieval."""
    original_question = state["question"]
    retries = state.get("retries", 0)

    prompt = f"""
Rewrite this question to improve retrieval from technical documentation.

Original question:
{original_question}

Return only the rewritten question.
Keep the original intent. Use alternative technical terms
and relevant concepts where appropriate.
"""

    try:
        response = llm.invoke(prompt)
        rewritten = response.content.strip()

        if not rewritten:
            rewritten = original_question

        state["rewritten_question"] = rewritten

    except Exception as exc:
        print(f"Query rewriting failed: {exc}")
        state["rewritten_question"] = original_question

    state["retries"] = retries + 1

    print(f"Rewritten query (attempt {state['retries']}): "
          f"{state['rewritten_question']}")

    return state



def generate_answer(state):
    docs = state.get("filtered_documents", [])

    if not docs:
        state["sources"] = []
        state["answer"] = (
            "I couldn't find relevant information in the indexed documentation."
        )
        return state

    # Assign a citation number to each unique source URL.
    source_numbers = {}
    context_parts = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")

        if source not in source_numbers:
            source_numbers[source] = len(source_numbers) + 1

        citation_number = source_numbers[source]

        context_parts.append(
            f"[Source {citation_number}]\n"
            f"URL: {source}\n"
            f"Content:\n{doc.page_content}"
        )

    state["sources"] = list(source_numbers.keys())
    context = "\n\n".join(context_parts)

    prompt = f"""
You are an expert AI documentation assistant.

Answer the user's question using ONLY the documentation provided below.

Rules:
- Answer clearly and directly.
- Add numbered citations such as [1] or [2] to claims supported
  by the corresponding source.
- Use only citation numbers supplied in the documentation.
- Do not invent sources, URLs, or citations.
- If the documentation provides only partial information, explain
  what it supports without inventing missing details.
- If the documentation contains nothing relevant, say:
  "I couldn't find this information in the indexed documentation."

Documentation:
{context}

Question:
{state["question"]}

Answer:
"""

    try:
        response = llm.invoke(prompt)
        state["answer"] = response.content.strip()

    except Exception as exc:
        print(f"LLM Error: {exc}")
        state["answer"] = (
            "The language model is temporarily unavailable."
        )

    return state
