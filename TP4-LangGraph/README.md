# 🤖 Orchestration d'Agents avec LangGraph — TP4

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/framework-LangGraph-orange)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)](https://streamlit.io/)

Ce projet présente une exploration approfondie de l'orchestration d'agents intelligents à l'aide de **LangGraph**. Il couvre les concepts fondamentaux jusqu'à la mise en œuvre d'un système de **RAG Agentique** (Retrieval-Augmented Generation) capable de raisonnement autonome, d'utilisation d'outils et de correction de trajectoire.

---

## 🌟 Points Forts

- **Workflows Flexibles** : Implémentation de graphes d'états linéaires et cycliques.
- **RAG Agentique** : Système capable de valider la pertinence des documents et d'effectuer des recherches web complémentaires.
- **Interface Interactive** : Chatbot Streamlit complet avec visualisation des appels d'outils en temps réel.
- **Observabilité** : Intégration native avec LangSmith pour le tracing et le debugging.

---

## 🏗️ Architecture du Système

Le projet est structuré pour offrir une progression pédagogique :

### 1. Fondamentaux de LangGraph (`partie1_bases/`)
Apprentissage des primitives de LangGraph à travers trois exemples progressifs :
- **Traitement Linéaire** : Gestion de dossiers employés avec nœuds de décision.
- **Gestion des Cycles** : Boucles de validation de documents pour assurer la complétude des données.
- **Agent ReAct** : Premier agent autonome utilisant des outils externes (calculatrice, recherche).

### 2. RAG Agentique Avancé (`partie2_agentic_rag/`)
Un système sophistiqué de recherche documentaire qui ne se contente pas de récupérer des données, mais les évalue :
- **Node-based RAG** : Décomposition du processus en nœuds : `retrieve`, `grade`, `generate`, `rewrite`.
- **Self-Correction** : Si les documents récupérés sont hors-sujet, l'agent reformule la requête et interroge le web (via Tavily).

### 3. Application Web (`streamlit_app/`)
Une interface utilisateur moderne pour interagir avec l'agent RAG, offrant une visibilité totale sur le "raisonnement" de l'IA.

---

## 📁 Structure du Projet

```text
TP4-LangGraph/
├── partie1_bases/            # Concepts fondamentaux (Graphes, États, Nœuds)
├── partie2_agentic_rag/      # Système RAG complet
│   ├── graph/                # Définition du workflow (State, Nodes, Edges)
│   ├── tools/                # Outils métier (ChromaDB, Email, Info Entreprise)
│   ├── data/                 # Base de connaissances (Textes & PDFs)
│   └── utils/                # Utilitaires (VectorStore)
├── streamlit_app/            # Interface utilisateur Web
├── requirements.txt          # Dépendances du projet
└── .env                      # Configuration des clés API
```

---

## ⚙️ Installation & Configuration

### 1. Prérequis
- Python 3.11 ou supérieur
- Clé API OpenAI (modèle `gpt-4o-mini` recommandé)

### 2. Mise en route rapide

```bash
# Créez et activez l'environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate

# Installez les dépendances
pip install -r requirements.txt
```

### 3. Configuration de l'environnement
Créez un fichier `.env` à la racine (ou copiez `.env.example`) :
```env
OPENAI_API_KEY=votre_cle_openai
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=votre_cle_langsmith  # Optionnel
TAVILY_API_KEY=votre_cle_tavily        # Optionnel pour la recherche web
```

---

## 🚀 Utilisation

### Mode Démonstration (Streamlit)
La façon la plus intuitive d'explorer le projet :
```bash
streamlit run streamlit_app/app.py
```

### Exemples de Base
```bash
python partie1_bases/01_workflow_lineaire.py
```

### Backend RAG (CLI)
```bash
python partie2_agentic_rag/main.py
```

---

## 🛠️ Stack Technique

- **Orchestration** : LangGraph, LangChain
- **LLM** : OpenAI GPT-4o-mini
- **Vector Store** : ChromaDB
- **UI** : Streamlit
- **Outils** : Tavily (Web Search), PyPDF

---
*Projet réalisé dans le cadre du module Intelligence Artificielle & LLM *
