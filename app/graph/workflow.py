
from langgraph.graph import StateGraph, END

from app.graph.state import GraphState
from app.graph.nodes import (
    analyze_query,
    retrieve_documents,
    grade_documents,
    generate_answer,
    decide_after_grading,
    rewrite_query,
)

workflow = StateGraph(GraphState)

workflow.add_node("analyze", analyze_query)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("grade", grade_documents)
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("generate", generate_answer)

workflow.set_entry_point("analyze")

workflow.add_edge("analyze", "retrieve")
workflow.add_edge("retrieve", "grade")

workflow.add_conditional_edges(
    "grade",
    decide_after_grading,
    {
        "generate": "generate",
        "rewrite": "rewrite",
    },
)

workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

graph = workflow.compile()
