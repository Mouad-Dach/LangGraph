"""
Partie 2 — Agentic RAG : Initialisation du VectorStore ChromaDB
================================================================
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Chemin de persistance ChromaDB
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
COLLECTION_NAME = "tp4_rag_collection"
DATA_DIR = Path(__file__).parent.parent / "data"

# Instance globale (singleton)
_vectorstore = None


def _load_documents():
    """Charge les documents depuis le dossier data/."""
    documents = []
    for file_path in DATA_DIR.glob("*.txt"):
        print(f"   📄 Chargement : {file_path.name}")
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents.extend(loader.load())

    if not documents:
        print("   ⚠️  Aucun document trouvé dans data/ — création d'un document exemple")
        from langchain_core.documents import Document
        documents = [
            Document(
                page_content="LangGraph est un framework pour créer des applications LLM avec des graphes d'état. Il permet de construire des agents complexes avec des boucles et des branches conditionnelles.",
                metadata={"source": "default"}
            )
        ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)
    print(f"   → {len(chunks)} chunks créés depuis {len(documents)} documents")
    return chunks


def get_vectorstore() -> Chroma:
    """Retourne le vectorstore ChromaDB (initialise si nécessaire)."""
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Si la DB existe déjà, on la charge
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        print("🗄️  Chargement du vectorstore existant...")
        _vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
        )
    else:
        print("🔨 Création du vectorstore ChromaDB...")
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        chunks = _load_documents()
        _vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(CHROMA_DIR),
            collection_name=COLLECTION_NAME,
        )
        print("   ✅ VectorStore créé et persisté")

    return _vectorstore


def get_retriever(k: int = 4):
    """Retourne un retriever configuré."""
    vs = get_vectorstore()
    return vs.as_retriever(search_kwargs={"k": k})


def reset_vectorstore():
    """Supprime et recrée le vectorstore (utile pour les tests)."""
    global _vectorstore
    import shutil
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print("🗑️  VectorStore supprimé")
    _vectorstore = None
    return get_vectorstore()
