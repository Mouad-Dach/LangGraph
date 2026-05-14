"""
Partie 1 — Exemple 2 : Graphe avec branches conditionnelles
=============================================================
Concepts : add_conditional_edges, routage dynamique
"""

from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END


# ─── État ────────────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    question: str
    category: str      # "math", "history", ou "unknown"
    answer: str


# ─── Nœud de classification ───────────────────────────────────────────────────
def classify_question(state: GraphState) -> GraphState:
    """Classe la question dans une catégorie."""
    q = state["question"].lower()
    if any(word in q for word in ["calcul", "addition", "soustraction", "nombre", "math"]):
        category = "math"
    elif any(word in q for word in ["histoire", "date", "guerre", "roi", "révolution"]):
        category = "history"
    else:
        category = "unknown"

    print(f"🔍 Classification : '{state['question']}' → catégorie = '{category}'")
    return {"question": state["question"], "category": category, "answer": ""}


# ─── Routeur conditionnel ─────────────────────────────────────────────────────
def route_question(state: GraphState) -> Literal["answer_math", "answer_history", "answer_unknown"]:
    """Décide vers quel nœud diriger selon la catégorie."""
    category = state["category"]
    if category == "math":
        return "answer_math"
    elif category == "history":
        return "answer_history"
    else:
        return "answer_unknown"


# ─── Nœuds de réponse ─────────────────────────────────────────────────────────
def answer_math(state: GraphState) -> GraphState:
    print("🧮 Nœud MATH activé")
    return {**state, "answer": "📐 Je suis spécialisé en mathématiques ! Posez-moi votre calcul."}


def answer_history(state: GraphState) -> GraphState:
    print("📜 Nœud HISTOIRE activé")
    return {**state, "answer": "🏛️  Je suis spécialisé en histoire ! Posez-moi votre question historique."}


def answer_unknown(state: GraphState) -> GraphState:
    print("❓ Nœud INCONNU activé")
    return {**state, "answer": "🤷 Je ne sais pas dans quelle catégorie classer votre question."}


# ─── Construction du graphe ───────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(GraphState)

    # Nœuds
    builder.add_node("classify", classify_question)
    builder.add_node("answer_math", answer_math)
    builder.add_node("answer_history", answer_history)
    builder.add_node("answer_unknown", answer_unknown)

    # Flux principal
    builder.add_edge(START, "classify")

    # Branchement conditionnel
    builder.add_conditional_edges(
        "classify",          # nœud source
        route_question,      # fonction de routage
        {                    # mapping résultat → nœud
            "answer_math": "answer_math",
            "answer_history": "answer_history",
            "answer_unknown": "answer_unknown",
        }
    )

    # Toutes les branches finissent à END
    builder.add_edge("answer_math", END)
    builder.add_edge("answer_history", END)
    builder.add_edge("answer_unknown", END)

    return builder.compile()


# ─── Exécution ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  LangGraph — Exemple 2 : Graphe Conditionnel")
    print("=" * 55)

    graph = build_graph()

    print("\n📊 Structure du graphe :")
    graph.get_graph().print_ascii()

    # Tests avec différentes questions
    questions = [
        "Quel est le résultat de ce calcul mathématique ?",
        "Quelle est la date de la Révolution Française en histoire ?",
        "Quel temps fait-il aujourd'hui ?",
    ]

    for q in questions:
        print(f"\n{'─' * 50}")
        result = graph.invoke({"question": q, "category": "", "answer": ""})
        print(f"❓ Question : {result['question']}")
        print(f"📂 Catégorie : {result['category']}")
        print(f"💬 Réponse   : {result['answer']}")
