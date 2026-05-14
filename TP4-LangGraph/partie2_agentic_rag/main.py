"""
Partie 2 — Agentic RAG : Point d'entrée CLI
============================================
Lance le chatbot en mode ligne de commande.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Vérification des variables d'environnement
if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERREUR : La variable OPENAI_API_KEY n'est pas définie.")
    print("   Copie .env.example en .env et remplis ta clé API.")
    sys.exit(1)


def main():
    from graph.graph import build_rag_graph
    from utils.vectorstore import get_vectorstore

    print("=" * 60)
    print("  🤖 Agentic RAG Chatbot — LangGraph")
    print("=" * 60)

    # Initialisation du vectorstore
    print("\n📦 Initialisation du VectorStore...")
    get_vectorstore()
    print("✅ VectorStore prêt\n")

    # Construction du graphe
    print("🔨 Construction du graphe LangGraph...")
    graph = build_rag_graph()
    print("✅ Graphe prêt\n")

    # Afficher la structure
    print("📊 Structure du graphe :")
    try:
        graph.get_graph().print_ascii()
    except Exception:
        print("   (visualisation ASCII non disponible)")

    print("\n" + "=" * 60)
    print("  Posez vos questions (tapez 'exit' pour quitter)")
    print("=" * 60 + "\n")

    config = {"configurable": {"thread_id": "cli-session-001"}}

    while True:
        question = input("❓ Votre question : ").strip()

        if question.lower() in ["exit", "quit", "q"]:
            print("\n👋 Au revoir !")
            break

        if not question:
            continue

        print("\n⏳ Traitement en cours...\n")

        initial_state = {
            "messages": [],
            "question": question,
            "documents": [],
            "generation": "",
            "web_search": False,
            "iterations": 0,
        }

        try:
            result = graph.invoke(initial_state, config=config)

            print("\n" + "─" * 60)
            print("💬 RÉPONSE :")
            print("─" * 60)
            print(result["generation"])

            # Afficher les sources
            if result.get("documents"):
                print("\n📚 SOURCES UTILISÉES :")
                for i, doc in enumerate(result["documents"], 1):
                    source = doc.metadata.get("source", "inconnu")
                    doc_type = doc.metadata.get("type", "vectorstore")
                    print(f"  {i}. [{doc_type}] {source}")

            print("─" * 60 + "\n")

        except Exception as e:
            print(f"\n❌ Erreur : {e}")
            print("Vérifiez votre clé API et votre connexion internet.\n")


if __name__ == "__main__":
    # Ajouter le répertoire parent au path pour les imports absolus
    current_dir = Path(__file__).parent.resolve()
    project_root = current_dir.parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    main()
