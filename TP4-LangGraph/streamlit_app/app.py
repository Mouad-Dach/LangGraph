"""
Interface Web Streamlit — Agentic RAG Chatbot
=============================================
Approche Prof. Mohamed YOUSSFI :
- Agent create_react_agent avec outils (retriever, company info, email)
- Historique de conversation
- Affichage des outils appelés
- LangSmith tracing intégré

Lancer : streamlit run streamlit_app/app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st

# Ajout des chemins au sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "partie2_agentic_rag"))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# ─── Configuration Page ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG — LangGraph",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Correction du chemin d'importation pour Streamlit
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from partie2_agentic_rag.tools.tools import all_tools, get_vectorstore

st.markdown("""
<style>
    .tool-call {
        background: #1a2744;
        border-left: 3px solid #4a9eff;
        padding: 8px 12px;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.85rem;
        color: #90cdf4;
        margin: 4px 0;
    }
    .tool-result {
        background: #1a3322;
        border-left: 3px solid #48bb78;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.8rem;
        color: #9ae6b4;
        margin: 4px 0;
    }
    .step-header { font-weight: bold; color: #e2e8f0; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "thread_id" not in st.session_state:
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())[:8]


# ─── Chargement de l'agent ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_agent():
    try:
        from partie2_agentic_rag.tools.tools import get_vectorstore
        get_vectorstore()
        from partie2_agentic_rag.agent import create_agent
        agent = create_agent()
        return agent, None
    except Exception as e:
        import traceback
        return None, f"{str(e)}\n{traceback.format_exc()}"


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Agentic RAG")
    st.markdown("**Prof. Mohamed YOUSSFI**")
    st.markdown("---")

    # Status API
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key.startswith("sk-"):
        st.success("✅ OpenAI configuré")
    else:
        st.error("❌ OPENAI_API_KEY manquante")
        user_key = st.text_input("Clé API OpenAI", type="password")
        if user_key:
            os.environ["OPENAI_API_KEY"] = user_key
            st.rerun()

    # LangSmith status
    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        st.success("✅ LangSmith tracing ON")
    else:
        st.info("ℹ️ LangSmith désactivé")

    st.markdown("---")

    st.markdown("### 🛠️ Outils disponibles")
    st.markdown("""
    | Outil | Description |
    |-------|-------------|
    | 🔍 `retriever_tool` | Base vectorielle ChromaDB |
    | 🏢 `get_company_info_tool` | Infos entreprises |
    | 📧 `send_email_tool` | Envoi d'emails |
    """)

    st.markdown("---")

    st.markdown("### 🏗️ Architecture ReAct")
    st.markdown("""
    ```
    START
      │
    [LLM] ──appel outil──► [tools]
      ▲                        │
      └────────résultat────────┘
      │
    (fin) ──► END
    ```
    """)

    st.markdown("---")
    st.markdown(f"**Session :** `{st.session_state.thread_id}`")

    if st.button("🗑️ Nouvelle conversation", use_container_width=True):
        import uuid
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.cache_resource.clear()
        st.rerun()


# ─── Zone principale ──────────────────────────────────────────────────────────
st.title("🤖 Agentic RAG Chatbot")
st.caption("LangGraph + ChromaDB + OpenAI · Inspiré du cours Prof. YOUSSFI")

if not os.getenv("OPENAI_API_KEY"):
    st.warning("⚠️ Configurez votre clé API dans la sidebar.")
    st.stop()

with st.spinner("⏳ Chargement de l'agent et du VectorStore..."):
    agent, error = load_agent()

if error:
    st.error(f"❌ Erreur : {error}")
    st.stop()

# Message de bienvenue
if not st.session_state.messages:
    st.info("""
    👋 **Bienvenue !** Cet agent peut :
    - 📚 Répondre aux questions sur les documents internes (RAG)
    - 🏢 Fournir des infos sur des entreprises
    - 📧 Simuler l'envoi d'emails
    - 🔄 Décomposer des questions complexes en plusieurs recherches

    **Essayez :** *"Quel est le chiffre d'affaires de Tech Maroc ?"*
    ou *"Explique-moi le RAG agentique"*
    """)

# Historique des messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            with st.expander("🔧 Outils utilisés", expanded=False):
                for tc in msg["tool_calls"]:
                    st.markdown(
                        f'<div class="tool-call">🔧 <b>{tc["tool"]}</b>({tc["input"]})</div>',
                        unsafe_allow_html=True
                    )
                    if tc.get("output"):
                        preview = tc["output"][:200] + "..." if len(tc["output"]) > 200 else tc["output"]
                        st.markdown(
                            f'<div class="tool-result">↩ {preview}</div>',
                            unsafe_allow_html=True
                        )


# ─── Saisie utilisateur ───────────────────────────────────────────────────────
if prompt := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🤔 L'agent réfléchit et cherche..."):
            try:
                from langchain_core.messages import HumanMessage

                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                result = agent.invoke(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config,
                )

                # Extraire la réponse finale
                final_answer = result["messages"][-1].content

                # Extraire les appels d'outils depuis les messages intermédiaires
                tool_calls = []
                messages = result.get("messages", [])
                for i, m in enumerate(messages):
                    # Message avec tool_calls (appel)
                    if hasattr(m, "tool_calls") and m.tool_calls:
                        for tc in m.tool_calls:
                            entry = {
                                "tool": tc.get("name", ""),
                                "input": str(tc.get("args", "")),
                                "output": "",
                            }
                            # Chercher le résultat dans le message suivant
                            if i + 1 < len(messages):
                                next_m = messages[i + 1]
                                if hasattr(next_m, "content"):
                                    entry["output"] = str(next_m.content)
                            tool_calls.append(entry)

                # Afficher la réponse
                st.markdown(final_answer)

                # Afficher les outils
                if tool_calls:
                    with st.expander(f"🔧 {len(tool_calls)} outil(s) utilisé(s)", expanded=False):
                        for tc in tool_calls:
                            st.markdown(
                                f'<div class="tool-call">🔧 <b>{tc["tool"]}</b>({tc["input"][:80]})</div>',
                                unsafe_allow_html=True
                            )
                            if tc.get("output"):
                                preview = tc["output"][:200] + "..." if len(tc["output"]) > 200 else tc["output"]
                                st.markdown(
                                    f'<div class="tool-result">↩ {preview}</div>',
                                    unsafe_allow_html=True
                                )

                # Sauvegarder
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_answer,
                    "tool_calls": tool_calls,
                })

            except Exception as e:
                err = f"❌ Erreur : {str(e)}"
                st.error(err)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err,
                    "tool_calls": [],
                })
