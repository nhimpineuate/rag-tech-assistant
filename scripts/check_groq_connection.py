"""Test the configured LLM connection."""

from app.llm import llm
from app.config import MODEL_NAME

print(f"Testing model: {MODEL_NAME}")

try:
    response = llm.invoke("Explain RAG in two sentences.")
    print("\nLLM connection successful.")
    print(response.content)

except Exception as exc:
    print("\nLLM connection failed.")
    print(f"Error type: {type(exc).__name__}")
    print(f"Error: {exc}")
