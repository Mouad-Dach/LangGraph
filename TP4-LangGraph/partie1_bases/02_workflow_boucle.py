"""
Partie 1 — Exemple 2 : Workflow avec Boucle
=============================================
Cas : Vérification des documents manquants d'un employé.
Le système boucle entre "notification" et "vérification"
jusqu'à ce que le dossier soit complet.

Concepts : boucles (cycles), arêtes conditionnelles,
           compteur d'itérations, MemorySaver
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver


# ─── État ────────────────────────────────────────────────────────────────────
class DossierState(TypedDict):
    nom_employe: str
    documents_requis: List[str]
    documents_fournis: List[str]
    documents_manquants: List[str]
    nb_relances: int
    max_relances: int
    statut: str      # "incomplet" | "complet" | "abandonne"
    historique: List[str]


# ─── Nœuds ───────────────────────────────────────────────────────────────────
def verifier_dossier(state: DossierState) -> DossierState:
    """Vérifie quels documents sont manquants."""
    print(f"\n🔍 Vérification du dossier de {state['nom_employe']}...")

    manquants = [
        doc for doc in state["documents_requis"]
        if doc not in state["documents_fournis"]
    ]

    statut = "complet" if not manquants else "incomplet"
    log = f"Vérification #{state['nb_relances']} : {len(manquants)} document(s) manquant(s)"

    print(f"   → Manquants : {manquants if manquants else 'Aucun ✅'}")

    return {
        **state,
        "documents_manquants": manquants,
        "statut": statut,
        "historique": state["historique"] + [log],
    }


def notifier_employe(state: DossierState) -> DossierState:
    """Notifie l'employé des documents manquants."""
    relance = state["nb_relances"] + 1
    msg = (
        f"📧 Relance #{relance} envoyée à {state['nom_employe']} — "
        f"Documents manquants : {', '.join(state['documents_manquants'])}"
    )
    print(f"   {msg}")

    # Simulation : l'employé fournit un document à chaque relance
    docs_simules = state["documents_fournis"].copy()
    if state["documents_manquants"]:
        doc_fourni = state["documents_manquants"][0]
        docs_simules.append(doc_fourni)
        print(f"   📨 Simulation : l'employé a fourni '{doc_fourni}'")

    return {
        **state,
        "documents_fournis": docs_simules,
        "nb_relances": relance,
        "historique": state["historique"] + [msg],
    }


def cloturer_dossier(state: DossierState) -> DossierState:
    """Clôture le dossier (complet ou abandonné)."""
    if state["statut"] == "complet":
        msg = f"✅ Dossier de {state['nom_employe']} COMPLET — Traitement lancé."
    else:
        msg = (
            f"⛔ Dossier de {state['nom_employe']} ABANDONNÉ après "
            f"{state['nb_relances']} relances. Documents manquants : "
            f"{', '.join(state['documents_manquants'])}"
        )
    print(f"\n{msg}")
    return {**state, "historique": state["historique"] + [msg]}


# ─── Routeur ─────────────────────────────────────────────────────────────────
def router_dossier(state: DossierState) -> str:
    """Décide si on continue la boucle, on abandonne, ou on clôture."""
    if state["statut"] == "complet":
        return "cloturer"
    if state["nb_relances"] >= state["max_relances"]:
        # Forcer statut abandonné
        state["statut"] = "abandonne"
        return "cloturer"
    return "notifier"


# ─── Graphe ───────────────────────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(DossierState)

    builder.add_node("verifier_dossier",  verifier_dossier)
    builder.add_node("notifier_employe",  notifier_employe)
    builder.add_node("cloturer_dossier",  cloturer_dossier)

    builder.add_edge(START, "verifier_dossier")

    builder.add_conditional_edges(
        "verifier_dossier",
        router_dossier,
        {
            "notifier":  "notifier_employe",
            "cloturer":  "cloturer_dossier",
        }
    )

    # La boucle : après notification → re-vérification
    builder.add_edge("notifier_employe", "verifier_dossier")
    builder.add_edge("cloturer_dossier", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# ─── Test ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  LangGraph — Exemple 2 : Workflow avec Boucle")
    print("=" * 55)

    graph = build_graph()

    print("\n📊 Structure du graphe :")
    graph.get_graph().print_ascii()

    # Cas 1 : dossier presque complet (2 docs manquants, max 3 relances)
    print("\n" + "═"*55)
    print("  CAS 1 : Dossier presque complet")
    print("═"*55)

    state1 = {
        "nom_employe": "Youssef Alami",
        "documents_requis": ["CIN", "Diplôme", "Photo", "RIB"],
        "documents_fournis": ["CIN", "Photo"],
        "documents_manquants": [],
        "nb_relances": 0,
        "max_relances": 3,
        "statut": "incomplet",
        "historique": [],
    }

    config1 = {"configurable": {"thread_id": "dossier-001"}}
    result1 = graph.invoke(state1, config=config1)
    print(f"\n📜 Historique :")
    for log in result1["historique"]:
        print(f"   • {log}")

    # Cas 2 : dossier avec trop de docs manquants (abandon)
    print("\n" + "═"*55)
    print("  CAS 2 : Trop de documents manquants → Abandon")
    print("═"*55)

    state2 = {
        "nom_employe": "Nadia Tahiri",
        "documents_requis": ["CIN", "Diplôme", "Photo", "RIB", "Attestation"],
        "documents_fournis": [],
        "documents_manquants": [],
        "nb_relances": 0,
        "max_relances": 2,
        "statut": "incomplet",
        "historique": [],
    }

    config2 = {"configurable": {"thread_id": "dossier-002"}}
    result2 = graph.invoke(state2, config=config2)
    print(f"\n📜 Historique :")
    for log in result2["historique"]:
        print(f"   • {log}")
