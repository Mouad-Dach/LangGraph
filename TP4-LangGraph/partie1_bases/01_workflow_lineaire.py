"""
Partie 1 — Exemple 1 : Workflow Linéaire
==========================================
Cas : Analyse d'un dossier employé avec des décisions
basées sur des règles métier (âge, salaire).

Concepts : StateGraph, TypedDict, nodes, edges, START, END
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# ─── 1. État partagé ──────────────────────────────────────────────────────────
class EmployeState(TypedDict):
    nom: str
    age: int
    salaire: float
    anciennete: int       # en années
    decision: str
    message: str


# ─── 2. Nœuds ────────────────────────────────────────────────────────────────
def verifier_eligibilite(state: EmployeState) -> EmployeState:
    """Vérifie si l'employé est éligible à une prime."""
    print(f"📋 Vérification éligibilité pour : {state['nom']}")
    eligible = state["anciennete"] >= 2 and state["salaire"] < 8000
    return {**state, "decision": "eligible" if eligible else "non_eligible"}


def calculer_prime(state: EmployeState) -> EmployeState:
    """Calcule la prime selon l'ancienneté et le salaire."""
    print(f"💰 Calcul de la prime pour : {state['nom']}")
    taux = 0.10 if state["anciennete"] < 5 else 0.15
    prime = state["salaire"] * taux
    return {
        **state,
        "message": f"✅ Prime accordée : {prime:.2f} MAD (taux {taux*100:.0f}%)"
    }


def refuser_prime(state: EmployeState) -> EmployeState:
    """Refuse la prime et explique pourquoi."""
    print(f"❌ Refus pour : {state['nom']}")
    raison = []
    if state["anciennete"] < 2:
        raison.append("ancienneté insuffisante (< 2 ans)")
    if state["salaire"] >= 8000:
        raison.append("salaire trop élevé (>= 8000 MAD)")
    return {
        **state,
        "message": f"❌ Prime refusée — Raison(s) : {', '.join(raison)}"
    }


def notifier_employe(state: EmployeState) -> EmployeState:
    """Notifie l'employé du résultat."""
    print(f"📧 Notification envoyée à : {state['nom']}")
    return {
        **state,
        "message": state["message"] + f"\n📧 Notification envoyée à {state['nom']}."
    }


# ─── 3. Fonction de routage conditionnel ────────────────────────────────────
def router_decision(state: EmployeState) -> str:
    """Dirige vers calculer_prime ou refuser_prime selon la décision."""
    return state["decision"]   # "eligible" ou "non_eligible"


# ─── 4. Construction du graphe ───────────────────────────────────────────────
def build_graph():
    builder = StateGraph(EmployeState)

    # Nœuds
    builder.add_node("verifier_eligibilite", verifier_eligibilite)
    builder.add_node("calculer_prime", calculer_prime)
    builder.add_node("refuser_prime", refuser_prime)
    builder.add_node("notifier_employe", notifier_employe)

    # Flux
    builder.add_edge(START, "verifier_eligibilite")

    # Branchement conditionnel
    builder.add_conditional_edges(
        "verifier_eligibilite",
        router_decision,
        {
            "eligible":     "calculer_prime",
            "non_eligible": "refuser_prime",
        }
    )

    builder.add_edge("calculer_prime",  "notifier_employe")
    builder.add_edge("refuser_prime",   "notifier_employe")
    builder.add_edge("notifier_employe", END)

    return builder.compile()


# ─── 5. Test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  LangGraph — Exemple 1 : Workflow Linéaire Employé")
    print("=" * 55)

    graph = build_graph()

    print("\n📊 Structure du graphe :")
    graph.get_graph().print_ascii()

    employes = [
        {"nom": "Ali Benali",    "age": 35, "salaire": 5500.0, "anciennete": 4, "decision": "", "message": ""},
        {"nom": "Sara Moussaoui","age": 28, "salaire": 9000.0, "anciennete": 3, "decision": "", "message": ""},
        {"nom": "Karim Idrissi", "age": 25, "salaire": 4000.0, "anciennete": 1, "decision": "", "message": ""},
    ]

    for emp in employes:
        print(f"\n{'─'*50}")
        print(f"👤 Traitement de : {emp['nom']}")
        result = graph.invoke(emp)
        print(result["message"])
