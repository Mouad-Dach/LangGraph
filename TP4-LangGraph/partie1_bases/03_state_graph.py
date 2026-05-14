"""
Partie 1 — Exemple 3 : Graphe avec état partagé + LLM (ChatOpenAI)
====================================================================
Concepts : état avec messages, intégration LLM, MemorySaver (checkpointing)
"""

import os
from typing import Annotated
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# ─── État avec réducteur de messages ─────────────────────────────────────────
class ChatState(TypedDict):
    # add_messages est un réducteur : il AJOUTE les nouveaux messages
    # au lieu de remplacer la liste entière
    messages: Annotated[list[BaseMessage], add_messages]
    turn_count: int


# ─── Initialisation du LLM ───────────────────────────────────────────────────
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0.7,
)


# ─── Nœud LLM ────────────────────────────────────────────────────────────────
def chat_node(state: ChatState) -> ChatState:
    """Appelle le LLM avec l'historique des messages."""
    print(f"🤖 LLM appelé — tour #{state['turn_count'] + 1}")
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],   # add_messages AJOUTE ce message
        "turn_count": state["turn_count"] + 1,
    }


# ─── Construction du graphe ───────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(ChatState)

    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)

    # MemorySaver = mémoire en RAM (persistance entre les appels du même thread)
    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


# ─── Exécution interactive ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  LangGraph — Exemple 3 : Chat avec Mémoire")
    print("=" * 55)

    graph = build_graph()

    print("\n📊 Structure du graphe :")
    graph.get_graph().print_ascii()

    # thread_id = identifiant de la conversation (mémoire isolée par thread)
    config = {"configurable": {"thread_id": "session-001"}}

    print("\n💬 Mode chat activé (tapez 'exit' pour quitter)\n")

    while True:
        user_input = input("Vous : ").strip()
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Au revoir !")
            break
        if not user_input:
            continue

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)], "turn_count": 0},
            config=config,
        )

        ai_message = result["messages"][-1]
        print(f"🤖 Assistant : {ai_message.content}\n")

        # Afficher le nombre de messages dans l'historique
        total_messages = len(result["messages"])
        print(f"   [Historique : {total_messages} messages — Tour {result['turn_count']}]\n")
