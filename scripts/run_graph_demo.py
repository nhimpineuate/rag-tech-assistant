
from app.graph.workflow import graph
import re

result = graph.invoke(
    {
        "question": "What is Agentic RAG?",
        "rewritten_question": "",
        "documents": [],
        "filtered_documents": [],
        "answer": "",
        "sources": [],
        "retries": 0,
    }
)

print("\nAnswer:\n")
print(result["answer"])

print("\nSources:")
for source in result.get("sources", []):
    print(f"- {source}")

# Validate citation numbers against available sources.
citation_numbers = set(
    re.findall(r"(?:\[(\d+)\]|【(\d+)】)", result["answer"])
)
citation_numbers = {
    a or b for a, b in citation_numbers
}

valid_numbers = {
    str(i)
    for i in range(1, len(result.get("sources", [])) + 1)
}

invalid_numbers = citation_numbers - valid_numbers

print("\nCitation validation:")
if not citation_numbers:
    print("WARNING: No numbered citations found.")
elif invalid_numbers:
    print(f"INVALID citation numbers: {sorted(invalid_numbers)}")
else:
    print("All citation numbers refer to an available source.")
