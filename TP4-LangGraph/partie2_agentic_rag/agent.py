"""
Partie 2 — Agentic RAG : Agent Principal
==========================================
Approche du Prof : create_react_agent (LangGraph prébuilti)
L'agent dispose de plusieurs outils et décide lui-même
quand et comment les utiliser (pattern ReAct).

Architecture :
    START → [assistant LLM] ⇄ [tools] → END
              ↑                    ↓
              └────────────────────┘
                  (boucle ReAct)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Ajout du chemin racine pour permettre les imports absolus
current_dir = Path(__file__).parent.resolve()
project_root = current_dir.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from partie2_agentic_rag.tools.tools import all_tools, get_vectorstore


# ─── Prompt système ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Tu es un assistant IA intelligent et polyvalent.
Tu disposes de plusieurs outils pour répondre aux questions :

1. retriever_tool     : pour rechercher dans la base de connaissances interne
                        (documents d'entreprise, guides techniques, politiques RH)
2. get_company_info   : pour obtenir des informations sur une entreprise par son nom
3. send_email         : pour envoyer des notifications par email

Instructions :
- Utilise TOUJOURS retriever_tool pour les questions sur les documents internes
- Pour des questions complexes, tu peux appeler retriever_tool PLUSIEURS FOIS
  avec des requêtes différentes pour reconstruire un contexte complet
- Sois précis, structuré et cite tes sources
- Si tu ne trouves pas l'information, dis-le clairement
- Réponds en français
"""


# ─── Création de l'agent ──────────────────────────────────────────────────────
def create_agent():
    """Crée et retourne l'agent RAG avec create_react_agent."""
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )

    memory = MemorySaver()

    # create_react_agent de LangGraph prébuilti
    # Il crée automatiquement la boucle : assistant ↔ tools
    agent = create_react_agent(
        model=llm,
        tools=all_tools,
        checkpointer=memory,
        prompt=SYSTEM_PROMPT,
    )

    return agent


# ─── Fonction d'invocation ────────────────────────────────────────────────────
def ask_agent(agent, question: str, thread_id: str = "default") -> dict:
    """Envoie une question à l'agent et retourne le résultat."""
    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config=config,
    )

    return result


# ─── Mode CLI interactif ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  🤖 Agentic RAG Chatbot — LangGraph")
    print("  Prof. Mohamed YOUSSFI — Approche create_react_agent")
    print("=" * 60)

    # Initialisation
    print("\n⏳ Initialisation du VectorStore...")
    try:
        get_vectorstore()
        print("✅ VectorStore prêt\n")
    except Exception as e:
        print(f"⚠️  Erreur VectorStore : {e}")

    print("🔨 Création de l'agent...")
    agent = create_agent()
    print("✅ Agent prêt\n")

    # Afficher la structure
    try:
        print("📊 Structure du graphe :")
        agent.get_graph().print_ascii()
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("  Posez vos questions (tapez 'exit' pour quitter)")
    print("  Exemples :")
    print("  - 'Quel est le CA de Tech Maroc en 2024 ?'")
    print("  - 'Explique-moi le pattern ReAct dans LangGraph'")
    print("  - 'Quels sont les critères de prime pour les employés ?'")
    print("  - 'Envoie un email à rh@techmaroc.ma pour informer de ma question'")
    print("=" * 60 + "\n")

    thread_id = "cli-main-001"

    while True:
        question = input("❓ Vous : ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("\n👋 Au revoir !")
            break
        if not question:
            continue

        print("\n⏳ L'agent réfléchit...\n")

        try:
            result = ask_agent(agent, question, thread_id)
            final_answer = result["messages"][-1].content

            print("─" * 60)
            print("💬 RÉPONSE :")
            print("─" * 60)
            print(final_answer)
            print("─" * 60 + "\n")

        except Exception as e:
            print(f"❌ Erreur : {e}\n")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
