
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from app.graph.workflow import graph

app = FastAPI(
    title="RAG Tech Assistant",
    version="1.0.0",
    description="A documentation assistant powered by retrieval-augmented generation."
)


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The technical question to answer."
    )


@app.get("/")
def home():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty."
        )

    try:
        result = graph.invoke(
            {
                "question": question,
                "rewritten_question": "",
                "documents": [],
                "filtered_documents": [],
                "answer": "",
                "sources": [],
                "retries": 0,
            }
        )

        return {
            "question": question,
            "answer": result["answer"],
            "sources": result.get("sources", []),
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="The assistant encountered an internal error."
        )
