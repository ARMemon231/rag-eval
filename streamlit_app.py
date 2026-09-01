"""
streamlit_app.py — Streamlit UI for the LLM Evals Course TA chatbot.

Run with:
    streamlit run streamlit_app.py
"""

import time
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Evals — Course TA",
    page_icon="📒",
    layout="centered",
)

# ── Custom CSS for dark theme matching the reference UI ──────────────────
st.markdown(
    """
    <style>
    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] p {
        color: #e0e0e0 !important;
    }

    /* ── Chat message styling ── */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }

    /* ── Bottom note ── */
    .bottom-note {
        color: #888;
        font-size: 0.82rem;
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }

    /* ── Slider track colour ── */
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #e74c3c;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar: Retrieval settings ──────────────────────────────────────────
with st.sidebar:
    st.header("Retrieval settings")

    fetch_k = st.slider(
        "fetch_k — candidates from the vector store",
        min_value=1,
        max_value=50,
        value=10,
        help="How many candidates the bi-encoder retrieves before reranking.",
    )

    top_k = st.slider(
        "top_k — chunks kept after reranking",
        min_value=1,
        max_value=20,
        value=5,
        help="How many chunks survive reranking and are fed to the generator.",
    )

    st.divider()

    show_context = st.toggle("Show retrieved context", value=True)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.contexts = []
        st.rerun()

    st.markdown(
        "<br><br><p style='color:#888; font-size:0.8rem;'>"
        "Answers are grounded only in the course transcripts. "
        "If the material doesn't cover it, the assistant abstains.</p>",
        unsafe_allow_html=True,
    )


# ── Lazy-load the RAG pipeline (cached so it loads once) ────────────────
@st.cache_resource(show_spinner="Loading RAG pipeline …")
def get_pipeline(fk, tk):
    from src.rag_pipeline import RagPipeline

    return RagPipeline(fetch_k=fk, top_k=tk)


# ── Main area ────────────────────────────────────────────────────────────
st.title("📒 LLM Evals — Course TA")
st.caption("Ask anything about the course transcripts. Answers come only from the material.")

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "contexts" not in st.session_state:
    st.session_state.contexts = []

# Render chat history
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show the retrieved-context expander right after the assistant reply
        if (
            msg["role"] == "assistant"
            and show_context
            and idx < len(st.session_state.contexts)
            and st.session_state.contexts[idx]
        ):
            chunks = st.session_state.contexts[idx]
            with st.expander(f"📦 {len(chunks)} retrieved chunks"):
                for i, chunk in enumerate(chunks):
                    st.markdown(f"**Chunk {i + 1}**")
                    st.text(chunk[:500])
                    if i < len(chunks) - 1:
                        st.divider()


# ── Chat input ───────────────────────────────────────────────────────────
if user_input := st.chat_input(
    "e.g. What is the difference between online and offline eval?"
):
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.contexts.append(None)  # placeholder (user msgs have no context)

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            start = time.time()
            pipeline = get_pipeline(fetch_k, top_k)
            result = pipeline.invoke(user_input)
            elapsed = time.time() - start

        answer = result["answer"]
        context = result["context"]

        st.markdown(answer)

        # Show retrieved chunks if toggled on
        if show_context and context:
            with st.expander(f"📦 {len(context)} retrieved chunks · {elapsed:.1f}s"):
                for i, chunk in enumerate(context):
                    st.markdown(f"**Chunk {i + 1}**")
                    st.text(chunk[:500])
                    if i < len(context) - 1:
                        st.divider()

    # Save to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.contexts.append(context)
