"""
Partie 2 — Agentic RAG : Définition des outils de l'agent
===========================================================
Outils disponibles :
  1. retriever_tool  — recherche dans ChromaDB
  2. get_company_info_tool — infos sur une entreprise (simulation)
  3. send_email_tool — envoi d'email (simulation)
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

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool

load_dotenv()

# ─── Chemins ─────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent.parent / "data"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION = "tp4_rag"


# ─── Initialisation du VectorStore ───────────────────────────────────────────
def init_vectorstore() -> Chroma:
    """Charge ou crée le vectorstore ChromaDB à partir des documents."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print("🗄️  VectorStore existant chargé.")
        return Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_name=COLLECTION,
        )

    print("🔨 Création du VectorStore ChromaDB...")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    # Chargement des documents
    documents = []
    for txt_file in DATA_DIR.glob("*.txt"):
        print(f"   📄 Chargement : {txt_file.name}")
        loader = TextLoader(str(txt_file), encoding="utf-8")
        documents.extend(loader.load())

    for pdf_file in DATA_DIR.glob("*.pdf"):
        print(f"   📕 Chargement PDF : {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    if not documents:
        raise FileNotFoundError(f"Aucun document trouvé dans {DATA_DIR}")

    # Découpage
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"   → {len(chunks)} chunks générés depuis {len(documents)} documents")

    # Stockage
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION,
    )
    print("   ✅ VectorStore créé et persisté.")
    return vs


# ─── Outil 1 : Retriever Tool ─────────────────────────────────────────────────
_vectorstore = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = init_vectorstore()
    return _vectorstore


@tool
def retriever_tool(query: str) -> str:
    """
    Recherche dans la base de connaissances (documents internes).
    Utilise cet outil pour répondre aux questions sur l'entreprise,
    les guides techniques, les politiques RH, les produits, etc.
    """
    print(f"   🔍 retriever_tool invoqué avec : '{query}'")
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)

    if not docs:
        return "Aucun document pertinent trouvé dans la base de connaissances."

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "inconnu")
        results.append(f"[Document {i} — Source: {source}]\n{doc.page_content}")

    return "\n\n---\n\n".join(results)


# ─── Outil 2 : Informations Entreprise ───────────────────────────────────────
@tool
def get_company_info(company_name: str) -> str:
    """
    Récupère des informations générales sur une entreprise.
    Utilise cet outil quand l'utilisateur demande des infos sur
    une entreprise spécifique par son nom.
    """
    print(f"   🏢 get_company_info invoqué pour : '{company_name}'")

    # Simulation d'une API externe
    companies = {
        "tech maroc": {
            "nom": "Tech Maroc SA",
            "secteur": "Technologies de l'information",
            "fondation": 2010,
            "siège": "Casablanca, Maroc",
            "effectif": 450,
            "CA_2024": "120 millions MAD",
        },
        "attijariwafa": {
            "nom": "Attijariwafa Bank",
            "secteur": "Banque et services financiers",
            "fondation": 1904,
            "siège": "Casablanca, Maroc",
            "effectif": 21000,
        },
    }

    key = company_name.lower().strip()
    for k, v in companies.items():
        if k in key or key in k:
            lines = [f"{field}: {val}" for field, val in v.items()]
            return "Informations sur l'entreprise :\n" + "\n".join(lines)

    return f"Aucune information trouvée pour '{company_name}' dans la base."


# ─── Outil 3 : Envoi d'email ──────────────────────────────────────────────────
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """
    Envoie un email à un destinataire.
    Utilise cet outil quand l'utilisateur demande d'envoyer
    une notification ou un message par email.
    """
    print(f"   📧 send_email invoqué → À: {to}, Sujet: {subject}")

    # Simulation d'envoi (en production, utiliser smtplib ou une API email)
    log = (
        f"✅ Email envoyé avec succès !\n"
        f"   À       : {to}\n"
        f"   Sujet   : {subject}\n"
        f"   Message : {body[:100]}{'...' if len(body) > 100 else ''}"
    )
    return log


# Liste complète des outils
all_tools = [retriever_tool, get_company_info, send_email]
