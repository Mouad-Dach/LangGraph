"""
Partie 1 — Exemple 3 : Agent ReAct avec Outils
================================================
Cas : Agent LLM capable d'utiliser des outils (addition,
multiplication) en modélisant explicitement la boucle
entre l'assistant LLM et le nœud d'exécution des outils.

Concepts : ReAct loop, ToolNode, tools, bind_tools,
           messages reducer, MemorySaver
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict

load_dotenv()


# ─── 1. Définir les outils ────────────────────────────────────────────────────
@tool
def additionner(a: float, b: float) -> float:
    """Additionne deux nombres. Utilise cet outil pour toute addition."""
    print(f"   🔧 OUTIL additionner({a}, {b}) = {a + b}")
    return a + b


@tool
def multiplier(a: float, b: float) -> float:
    """Multiplie deux nombres. Utilise cet outil pour toute multiplication."""
    print(f"   🔧 OUTIL multiplier({a}, {b}) = {a * b}")
    return a * b


@tool
def soustraire(a: float, b: float) -> float:
    """Soustrait b de a. Utilise cet outil pour toute soustraction."""
    print(f"   🔧 OUTIL soustraire({a}, {b}) = {a - b}")
    return a - b


@tool
def diviser(a: float, b: float) -> float:
    """Divise a par b. Utilise cet outil pour toute division."""
    if b == 0:
        return "Erreur : division par zéro impossible"
    print(f"   🔧 OUTIL diviser({a}, {b}) = {a / b}")
    return a / b


# Liste des outils disponibles
tools = [additionner, multiplier, soustraire, diviser]


# ─── 2. État avec messages ────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ─── 3. LLM lié aux outils ───────────────────────────────────────────────────
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
)
llm_with_tools = llm.bind_tools(tools)


# ─── 4. Nœud LLM (assistant) ─────────────────────────────────────────────────
def assistant(state: AgentState) -> AgentState:
    """Le LLM analyse la question et décide d'utiliser un outil ou de répondre."""
    print("🤖 [LLM] Analyse de la question...")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# ─── 5. Construction du graphe ReAct ─────────────────────────────────────────
def build_graph():
    builder = StateGraph(AgentState)

    # Nœuds
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))   # ToolNode exécute les outils automatiquement

    # Flux
    builder.add_edge(START, "assistant")

    # tools_condition : si le LLM a appelé un outil → "tools", sinon → END
    builder.add_conditional_edges(
        "assistant",
        tools_condition,   # fonction prébuiltie LangGraph
    )

    # Après exécution de l'outil → retour au LLM (la boucle ReAct)
    builder.add_edge("tools", "assistant")

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# ─── 6. Test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  LangGraph — Exemple 3 : Agent ReAct avec Outils")
    print("=" * 55)

    graph = build_graph()

    print("\n📊 Structure du graphe ReAct :")
    graph.get_graph().print_ascii()

    config = {"configurable": {"thread_id": "react-session-001"}}

    questions = [
        "Calcule (15 + 27) × 3",
        "Si j'ai 150 MAD et que j'en dépense 47, puis que je multiplie le reste par 2, combien j'ai ?",
        "Divise 100 par 4, puis additionne 13 au résultat.",
    ]

    for question in questions:
        print(f"\n{'─'*55}")
        print(f"❓ Question : {question}")
        print("─"*55)

        result = graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config,
        )

        # La dernière réponse de l'assistant
        final_answer = result["messages"][-1].content
        print(f"\n💬 Réponse finale : {final_answer}")
