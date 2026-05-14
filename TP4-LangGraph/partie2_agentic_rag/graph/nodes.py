"""
Partie 2 — Agentic RAG : Définition des nœuds du graphe
=========================================================
Nœuds :
  - retrieve          : Récupère les documents du vectorstore
  - grade_documents   : Note la pertinence des documents
  - generate          : Génère la réponse finale
  - rewrite_query     : Reformule la question si besoin
  - web_search_node   : Effectue une recherche web (Tavily)
"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from .state import RAGState
from ..utils.vectorstore import get_retriever

load_dotenv()

# ─── Initialisation LLM ───────────────────────────────────────────────────────
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
)

llm_creative = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0.7,
)


# ─── Schéma Pydantic pour le grading ─────────────────────────────────────────
class GradeDocument(BaseModel):
    """Score binaire pour la pertinence d'un document."""
    score: Literal["yes", "no"] = Field(
        description="'yes' si le document est pertinent, 'no' sinon"
    )


# ─── 1. Nœud RETRIEVE ─────────────────────────────────────────────────────────
def retrieve(state: RAGState) -> RAGState:
    """Récupère les documents pertinents depuis ChromaDB."""
    print("📚 [NŒUD] retrieve — Recherche dans le vectorstore...")
    question = state["question"]
    retriever = get_retriever()
    documents = retriever.invoke(question)
    print(f"   → {len(documents)} documents récupérés")
    return {**state, "documents": documents, "iterations": state.get("iterations", 0)}


# ─── 2. Nœud GRADE DOCUMENTS ──────────────────────────────────────────────────
def grade_documents(state: RAGState) -> RAGState:
    """
    Évalue la pertinence de chaque document récupéré.
    Met web_search=True si des documents non pertinents sont trouvés.
    """
    print("🔎 [NŒUD] grade_documents — Évaluation de la pertinence...")
    question = state["question"]
    documents = state["documents"]

    grader_llm = llm.with_structured_output(GradeDocument)

    system_prompt = """Tu es un évaluateur de pertinence de documents pour une question donnée.
    Réponds uniquement avec 'yes' si le document contient des informations utiles pour répondre à la question,
    ou 'no' si le document n'est pas pertinent."""

    filtered_docs = []
    web_search_needed = False

    for i, doc in enumerate(documents):
        prompt = f"Question : {question}\n\nDocument : {doc.page_content}"
        result = grader_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ])
        if result.score == "yes":
            print(f"   ✅ Document {i+1} : pertinent")
            filtered_docs.append(doc)
        else:
            print(f"   ❌ Document {i+1} : non pertinent")
            web_search_needed = True

    if not filtered_docs:
        web_search_needed = True

    return {**state, "documents": filtered_docs, "web_search": web_search_needed}


# ─── 3. Nœud REWRITE QUERY ───────────────────────────────────────────────────
def rewrite_query(state: RAGState) -> RAGState:
    """Reformule la question pour améliorer la recherche."""
    print("✏️  [NŒUD] rewrite_query — Reformulation de la question...")
    question = state["question"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Tu es un expert en reformulation de questions pour améliorer 
        la recherche d'informations. Reformule la question pour la rendre plus précise 
        et efficace pour une recherche documentaire. Retourne uniquement la question reformulée."""),
        ("human", "Question originale : {question}"),
    ])

    chain = prompt | llm_creative | StrOutputParser()
    rewritten = chain.invoke({"question": question})
    print(f"   → Question reformulée : '{rewritten}'")

    return {**state, "question": rewritten, "iterations": state.get("iterations", 0) + 1}


# ─── 4. Nœud WEB SEARCH ──────────────────────────────────────────────────────
def web_search_node(state: RAGState) -> RAGState:
    """Effectue une recherche web via Tavily et ajoute les résultats aux documents."""
    print("🌐 [NŒUD] web_search — Recherche sur le web...")

    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        web_search_tool = TavilySearchResults(max_results=3)
        results = web_search_tool.invoke({"query": state["question"]})

        web_docs = []
        for r in results:
            web_docs.append(Document(
                page_content=r.get("content", ""),
                metadata={"source": r.get("url", "web"), "type": "web_search"}
            ))

        print(f"   → {len(web_docs)} résultats web récupérés")
        combined_docs = state["documents"] + web_docs
        return {**state, "documents": combined_docs}

    except Exception as e:
        print(f"   ⚠️  Recherche web impossible : {e}")
        print("   → Utilisation des documents existants uniquement")
        return state


# ─── 5. Nœud GENERATE ────────────────────────────────────────────────────────
def generate(state: RAGState) -> RAGState:
    """Génère la réponse finale à partir des documents filtrés."""
    print("💬 [NŒUD] generate — Génération de la réponse...")
    question = state["question"]
    documents = state["documents"]

    # Formater le contexte
    if documents:
        context = "\n\n---\n\n".join([
            f"Source : {doc.metadata.get('source', 'inconnu')}\n{doc.page_content}"
            for doc in documents
        ])
    else:
        context = "Aucun document pertinent trouvé."

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Tu es un assistant IA expert. Réponds à la question de l'utilisateur 
        en te basant sur le contexte fourni. Si le contexte ne contient pas suffisamment 
        d'informations, dis-le clairement. Sois précis, concis et structuré.
        
        Contexte :
        {context}"""),
        ("human", "{question}"),
    ])

    chain = prompt | llm_creative | StrOutputParser()
    generation = chain.invoke({"context": context, "question": question})

    print(f"   → Réponse générée ({len(generation)} caractères)")

    return {
        **state,
        "generation": generation,
        "messages": [
            HumanMessage(content=question),
            AIMessage(content=generation),
        ],
    }


# ─── Fonctions de routage conditionnel ───────────────────────────────────────
def decide_to_generate(state: RAGState) -> Literal["rewrite_query", "generate"]:
    """
    Décide si on génère directement ou si on reformule la question.
    Limite les itérations à 2 pour éviter les boucles infinies.
    """
    iterations = state.get("iterations", 0)
    web_search = state.get("web_search", False)

    if web_search and iterations < 2:
        print("🔀 [ROUTEUR] → Reformulation de la question nécessaire")
        return "rewrite_query"
    else:
        print("🔀 [ROUTEUR] → Génération de la réponse")
        return "generate"
