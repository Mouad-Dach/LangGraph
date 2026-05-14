"""
Partie 2 — Agentic RAG : Définition de l'état du graphe
=========================================================
"""

from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class RAGState(TypedDict):
    """
    État global qui circule entre tous les nœuds du graphe RAG.

    Attributs :
        messages     : Historique de la conversation (avec réducteur add_messages)
        question     : La question courante de l'utilisateur
        documents    : Documents récupérés depuis le vectorstore
        generation   : La réponse générée par le LLM
        web_search   : Flag pour déclencher la recherche web
        iterations   : Compteur pour éviter les boucles infinies
    """
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    documents: List[Document]
    generation: str
    web_search: bool
    iterations: int
