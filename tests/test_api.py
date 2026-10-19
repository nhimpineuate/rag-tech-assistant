import os
from types import SimpleNamespace

# Allow tests to import the app without a real API key.
os.environ.setdefault("GROQ_API_KEY", "test-key-for-unit-tests")
os.environ.setdefault("MODEL_NAME", "openai/gpt-oss-120b")

import app.main as api
from fastapi.testclient import TestClient

client = TestClient(api.app)


def test_root_redirects_to_docs():
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_answer_and_sources(monkeypatch):
    def fake_invoke(state):
        assert state["question"] == "What is Agentic RAG?"
        assert state["retries"] == 0

        return {
            "answer": "Agentic RAG uses an agent to manage retrieval [1].",
            "sources": ["https://example.com/agentic-rag"],
        }

    monkeypatch.setattr(
        api, "graph", SimpleNamespace(invoke=fake_invoke)
    )

    response = client.post(
        "/query",
        json={"question": "  What is Agentic RAG?  "},
    )

    assert response.status_code == 200
    assert response.json() == {
        "question": "What is Agentic RAG?",
        "answer": "Agentic RAG uses an agent to manage retrieval [1].",
        "sources": ["https://example.com/agentic-rag"],
    }


def test_whitespace_only_question_is_rejected():
    response = client.post("/query", json={"question": "   "})

    assert response.status_code == 422
    assert response.json()["detail"] == "Question cannot be empty."


def test_question_over_maximum_length_is_rejected():
    response = client.post("/query", json={"question": "x" * 2001})

    assert response.status_code == 422


def test_workflow_error_returns_generic_message(monkeypatch):
    def fake_invoke(state):
        raise RuntimeError("Internal details must not be exposed")

    monkeypatch.setattr(
        api, "graph", SimpleNamespace(invoke=fake_invoke)
    )

    response = client.post(
        "/query",
        json={"question": "What is Agentic RAG?"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "The assistant encountered an internal error."
    }
    assert "Internal details must not be exposed" not in response.text
