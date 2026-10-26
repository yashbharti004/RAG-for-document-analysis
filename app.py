"""
Streamlit chat interface for the University Policy Assistant.
"""

import streamlit as st
import requests

st.set_page_config(
    page_title="University Policy Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for cleaner readability and source cards
st.markdown(
    """
    <style>
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.25rem;
    }
    .app-subtitle {
        color: #4b5563;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        line-height: 1.45;
    }
    .source-header {
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.25rem;
    }
    .source-snippet {
        color: #334155;
        font-style: italic;
    }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULT_API_URL = "http://localhost:8000"

if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_url" not in st.session_state:
    st.session_state.api_url = DEFAULT_API_URL


def check_api_health(url: str) -> dict | None:
    try:
        res = requests.get(f"{url.rstrip('/')}/health", timeout=3)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None


def query_rag_api(question: str, top_k: int = 6) -> dict:
    url = f"{st.session_state.api_url.rstrip('/')}/ask"
    try:
        res = requests.post(
            url,
            json={"question": question, "top_k": top_k},
            timeout=120,
        )
        res.raise_for_status()
        return res.json()
    except requests.ConnectionError:
        return {
            "error": "Could not connect to FastAPI server. Ensure it is running (`python api.py`)."
        }
    except requests.HTTPError as exc:
        msg = f"HTTP {exc.response.status_code}"
        try:
            detail = exc.response.json().get("detail")
            if detail:
                msg = f"{msg}: {detail}"
        except Exception:
            pass
        return {"error": msg}
    except Exception as exc:
        return {"error": str(exc)}


def render_sources_accordion(sources: list):
    if not sources:
        return

    with st.expander(f"Cited Sources ({len(sources)})", expanded=False):
        for src in sources:
            source_file = src.get("source", "Unknown Document")
            page_num = src.get("page", "?")
            section = src.get("section", "")
            snippet = src.get("snippet", "")

            sec_label = f" • Section: {section}" if section else ""
            st.markdown(
                f"""
                <div class="source-box">
                    <div class="source-header">
                        📄 {source_file} (Page {page_num}){sec_label}
                    </div>
                    <div class="source-snippet">"{snippet}"</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# Sidebar controls and health check
with st.sidebar:
    st.subheader("Configuration")
    st.session_state.api_url = st.text_input(
        "API Base URL",
        value=st.session_state.api_url,
        help="Endpoint where FastAPI server is hosted",
    )
    top_k = st.slider("Context chunks to retrieve", min_value=1, max_value=12, value=6)

    st.markdown("---")

    health = check_api_health(st.session_state.api_url)
    if health:
        status_color = "green" if health.get("index_loaded") else "orange"
        st.markdown(
            f"**Backend Status:** <span style='color:{status_color}; font-weight:600;'>Online</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"Model: `{health.get('model', 'default')}`")
        st.caption(f"Retriever: `{health.get('retriever_mode', 'standard')}`")
    else:
        st.markdown(
            "**Backend Status:** <span style='color:red; font-weight:600;'>Offline</span>",
            unsafe_allow_html=True,
        )
        st.caption("Start the server: `python api.py`")

    st.markdown("---")

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.subheader("Sample Questions")
    sample_queries = [
        "What is the minimum attendance requirement?",
        "What are the exam re-evaluation rules?",
        "What is the hostel curfew timing?",
        "What GPA is needed for a merit scholarship?",
        "Explain the anti-ragging policy.",
    ]
    for sample in sample_queries:
        if st.button(sample, key=f"q_{hash(sample)}", use_container_width=True):
            st.session_state["_selected_sample"] = sample
            st.rerun()


# Main chat view
st.markdown('<div class="app-title">University Policy Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Ask questions about official academic regulations, hostel guidelines, and student codes.</div>',
    unsafe_allow_html=True,
)

# Display existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources_accordion(msg["sources"])

# Input handling
sample_query = st.session_state.pop("_selected_sample", None)
user_query = sample_query or st.chat_input("Ask a question about university policy...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents..."):
            response = query_rag_api(user_query, top_k=top_k)

        if "error" in response:
            error_msg = f"Error: {response['error']}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            answer = response.get("answer", "No answer returned.")
            sources = response.get("sources", [])

            st.markdown(answer)
            if sources:
                render_sources_accordion(sources)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
