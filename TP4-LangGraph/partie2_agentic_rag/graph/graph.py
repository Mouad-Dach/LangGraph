"""
Partie 2 — Agentic RAG : Construction et compilation du graphe LangGraph
=========================================================================

Architecture :
    START
      │
      ▼
   [retrieve]
      │
      ▼
   [grade_documents]
      │
      ├── (pertinents) ─────────────────────────────────► [generate] ──► END
      │
      └── (non pertinents) ──► [rewrite_query] ──► [web_search] ──► [generate] ──► END
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import RAGState
from .nodes import (
    retrieve,
    grade_documents,
    rewrite_query,
    web_search_node,
    generate,
    decide_to_generate,
)


def build_rag_graph():
    """Construit et compile le graphe Agentic RAG."""
    builder = StateGraph(RAGState)

    # ── Ajouter les nœuds ──────────────────────────────────────────────────
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade_documents", grade_documents)
    builder.add_node("rewrite_query", rewrite_query)
    builder.add_node("web_search", web_search_node)
    builder.add_node("generate", generate)

    # ── Flux principal ────────────────────────────────────────────────────
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade_documents")

    # ── Branchement conditionnel après le grading ─────────────────────────
    builder.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "rewrite_query": "rewrite_query",
            "generate": "generate",
        },
    )

    # ── Flux après reformulation ──────────────────────────────────────────
    builder.add_edge("rewrite_query", "web_search")
    builder.add_edge("web_search", "generate")

    # ── Fin ───────────────────────────────────────────────────────────────
    builder.add_edge("generate", END)

    # ── Compilation avec mémoire (pour LangGraph Studio) ──────────────────
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# Point d'entrée pour LangGraph Studio (référencé dans langgraph.json)
app = build_rag_graph()
