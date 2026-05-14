"""
Partie 1 — Exemple 1 : Premier graphe simple avec LangGraph
============================================================
Concepts : StateGraph, nodes, edges, START, END
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# ─── 1. Définir l'état partagé ───────────────────────────────────────────────
class GraphState(TypedDict):
    """L'état qui circule entre les nœuds du graphe."""
    message: str
    step_count: int


# ─── 2. Définir les nœuds (fonctions) ────────────────────────────────────────
def node_hello(state: GraphState) -> GraphState:
    """Nœud 1 : Ajoute un message de bienvenue."""
    print("📍 Nœud 'hello' exécuté")
    return {
        "message": f"Bonjour ! {state['message']}",
        "step_count": state["step_count"] + 1,
    }


def node_process(state: GraphState) -> GraphState:
    """Nœud 2 : Traite le message."""
    print("📍 Nœud 'process' exécuté")
    return {
        "message": state["message"].upper(),
        "step_count": state["step_count"] + 1,
    }


def node_goodbye(state: GraphState) -> GraphState:
    """Nœud 3 : Termine le traitement."""
    print("📍 Nœud 'goodbye' exécuté")
    return {
        "message": f"{state['message']} — Traitement terminé en {state['step_count']} étapes.",
        "step_count": state["step_count"] + 1,
    }


# ─── 3. Construire le graphe ──────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(GraphState)

    # Ajouter les nœuds
    builder.add_node("hello", node_hello)
    builder.add_node("process", node_process)
    builder.add_node("goodbye", node_goodbye)

    # Ajouter les arêtes (flux d'exécution)
    builder.add_edge(START, "hello")
    builder.add_edge("hello", "process")
    builder.add_edge("process", "goodbye")
    builder.add_edge("goodbye", END)

    # Compiler le graphe
    return builder.compile()


# ─── 4. Exécuter ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  LangGraph — Exemple 1 : Graphe Simple")
    print("=" * 50)

    graph = build_graph()

    # Afficher la structure du graphe
    print("\n📊 Structure du graphe :")
    graph.get_graph().print_ascii()

    # Lancer le graphe
    initial_state = {"message": "Mon premier graphe LangGraph!", "step_count": 0}
    print(f"\n▶️  État initial : {initial_state}")
    print("\n--- Exécution ---")

    result = graph.invoke(initial_state)

    print("\n--- Résultat Final ---")
    print(f"✅ Message : {result['message']}")
    print(f"✅ Étapes  : {result['step_count']}")
