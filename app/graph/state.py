
from typing import TypedDict, List
from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str
    rewritten_question: str
    documents: List[Document]
    filtered_documents: List[Document]
    answer: str
    sources: List[str]
    retries: int
