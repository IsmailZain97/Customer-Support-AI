import os
os.environ["USE_TORCH"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import os
import pickle
import streamlit as st
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

from database import fetch_all_tickets
from train_model import predict_sentiment, MODEL_PATH, VECTORIZER_PATH, METRICS_PATH
from rag_engine import RAGEngine

# ─── 1. Config ────────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_CHOICE = "llama-3.1-8b-instant"

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY missing — add it to your .env file and restart.")
    st.stop()

try:
    client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Groq client error: {e}")
    st.stop()

# ─── 2. Cached resource loading ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML artefacts ...")
def load_ml_artefacts():
    """Load the trained sklearn model + vectoriser once and cache them."""
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except FileNotFoundError:
        return None, None

@st.cache_resource(show_spinner="Initialising RAG engine ...")
def load_rag_engine() -> RAGEngine:
    """Build the RAGEngine once (embedding model + doc index) and cache it."""
    return RAGEngine()

@st.cache_data(show_spinner="Loading tickets from PostgreSQL ...")
def load_tickets() -> pd.DataFrame:
    return fetch_all_tickets()

ml_model, ml_vectorizer = load_ml_artefacts()
rag_engine = load_rag_engine()

# ─── 3. Helpers ───────────────────────────────────────────────────────────────
def sentiment_badge(label: str, confidence: float) -> str:
    """Returns an HTML badge string for the given sentiment label."""
    colour = {"POSITIVE": "#28a745", "NEGATIVE": "#dc3545"}.get(label, "#6c757d")
    return (
        f'<span style="background:{colour};color:white;padding:4px 12px;'
        f'border-radius:20px;font-weight:600;">'
        f'{"😊" if label == "POSITIVE" else "😞"} {label} ({confidence:.0%})</span>'
    )

def generate_llm_response(ticket_text: str, sentiment: str, rag_context: str) -> str:
    """
    Calls the Groq API with a sentiment-aware system prompt + RAG context.

    The system prompt changes tone based on the sentiment:
      - NEGATIVE: empathetic, apologetic, solution-focused
      - POSITIVE:  warm, enthusiastic, appreciative
    """
    if sentiment == "NEGATIVE":
        system_prompt = (
            "You are an empathetic customer resolution specialist. "
            "The customer is upset or frustrated. Your goal is to:\n"
            "  1. Apologise sincerely and acknowledge their frustration\n"
            "  2. Provide a clear, actionable solution grounded in the policy documents below\n"
            "  3. Be calm, professional, and specific — never give a generic copy-paste reply\n\n"
            f"Company Policy Documents (use these to answer):\n{rag_context}"
        )
    else:
        system_prompt = (
            "You are an enthusiastic brand ambassador. "
            "The customer is happy and satisfied. Your goal is to:\n"
            "  1. Thank them warmly and match their positive energy\n"
            "  2. Acknowledge their specific compliment using the policy documents below\n"
            "  3. Invite them to explore loyalty perks if applicable\n\n"
            f"Company Policy Documents (use these to answer):\n{rag_context}"
        )

    response = client.chat.completions.create(
        model=MODEL_CHOICE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Customer message:\n{ticket_text}\n\nWrite an official support reply:"},
        ],
        temperature=0.5,
        max_tokens=400,
    )
    return response.choices[0].message.content


st.set_page_config(page_title="Customer Support AI", page_icon="🎧", layout="wide")

with st.sidebar:
    st.title("🎧 Support AI")
    st.markdown("---")

    # System status
    st.markdown("**System Status**")
    model_status = "✅ Ready" if ml_model else "❌ Not trained"
    st.markdown(f"ML Model: {model_status}")
    st.markdown("RAG Engine: ✅ Ready")
    st.markdown(f"LLM (Groq): ✅ {MODEL_CHOICE}")

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "rb") as f:
            metrics = pickle.load(f)
        st.markdown("---")
        st.markdown("**Model Performance**")
        st.metric("Test Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
        st.metric("ROC-AUC",       f"{metrics['roc_auc']:.4f}")
        st.metric("CV Score",      f"{metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")


    if os.path.exists("confusion_matrix.png"):
        st.markdown("---")
        st.markdown("**Confusion Matrix**")
        st.image("confusion_matrix.png", use_container_width=True)

    if ml_model is None:
        st.warning("Run `python train_model.py` first to train the ML model.")


st.title("🛡️ Enterprise AI Support — ML + RAG Pipeline")
st.markdown(
    "**Pipeline:** PostgreSQL ticket → Local ML sentiment → "
    "Semantic RAG retrieval → Groq Llama 3.1 response"
)
st.markdown("---")

tab1, tab2 = st.tabs(["📂 Ticket Analyser", "💬 Live Chat"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: Ticket Analyser
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    try:
        df = load_tickets()
    except Exception as e:
        st.error(f"PostgreSQL connection failed: {e}")
        st.stop()

    col_left, col_right = st.columns([1, 1])


    with col_left:
        st.subheader("📋 Select a Ticket")

        # Optional category filter
        categories = ["All"] + sorted(df["category"].unique().tolist())
        selected_cat = st.selectbox("Filter by category", categories)

        filtered_df = df if selected_cat == "All" else df[df["category"] == selected_cat]
        selected_id = st.selectbox("Ticket ID", filtered_df["ticket_id"].tolist())

        ticket = filtered_df[filtered_df["ticket_id"] == selected_id].iloc[0]

        with st.container(border=True):
            st.markdown(f"**Customer:** {ticket['customer_name']}")
            st.markdown(f"**Category:** `{ticket['category']}` | **Status:** `{ticket['status']}`")
            st.markdown(f"**Rating:** {'⭐' * int(ticket['rating'])}")
            st.markdown(f"**Issue:**")
            st.info(ticket["issue_description"])

    # ── Right column: pipeline output ─────────────────────────────────────────
    with col_right:
        st.subheader("🤖 AI Pipeline Output")

        if st.button("Run Full Pipeline", type="primary", use_container_width=True):
            if ml_model is None:
                st.error("ML model not found. Run `python train_model.py` first.")
            else:
                issue_text = ticket["issue_description"]

                # ── Step 1: Local ML Sentiment ─────────────────────────────────
                with st.spinner("Step 1 — Running sentiment model ..."):
                    result = predict_sentiment(issue_text, ml_model, ml_vectorizer)

                st.markdown("**Step 1 — Local ML Sentiment Prediction**")
                st.markdown(
                    sentiment_badge(result["label"], result["confidence"]),
                    unsafe_allow_html=True
                )
                # Confidence bar
                st.progress(result["confidence"], text=f"Confidence: {result['confidence']:.0%}")
                with st.expander("📊 Full probability breakdown"):
                    col_a, col_b = st.columns(2)
                    col_a.metric("POSITIVE probability", f"{result['positive_prob']:.0%}")
                    col_b.metric("NEGATIVE probability", f"{result['negative_prob']:.0%}")

                st.markdown("---")

                # ── Step 2: RAG Retrieval ──────────────────────────────────────
                with st.spinner("Step 2 — Retrieving relevant policy documents ..."):
                    retrieved_docs = rag_engine.retrieve(issue_text, top_k=2)
                    rag_context    = rag_engine.get_context_string(issue_text, top_k=2)

                st.markdown("**Step 2 — RAG: Retrieved Policy Documents**")
                st.caption(
                    "These are retrieved by *semantic similarity* to the customer's message, "
                    "not by a hardcoded category key."
                )
                for doc in retrieved_docs:
                    with st.expander(
                        f"📄 {doc['title']}  —  similarity {doc['similarity']:.0%}"
                    ):
                        st.write(doc["content"])

                st.markdown("---")

                # ── Step 3: LLM Response ───────────────────────────────────────
                with st.spinner("Step 3 — Generating AI response via Groq ..."):
                    ai_reply = generate_llm_response(
                        issue_text, result["label"], rag_context
                    )

                st.markdown("**Step 3 — AI-Generated Support Reply**")
                st.success(ai_reply)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: Live Chat
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("💬 Live Customer Chat")
    st.caption(
        "Type any customer message. The same ML → RAG → LLM pipeline runs in real time."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["user"])
            st.markdown(
                sentiment_badge(entry["sentiment"]["label"], entry["sentiment"]["confidence"]),
                unsafe_allow_html=True,
            )
        with st.chat_message("assistant"):
            st.write(entry["reply"])

    # Chat input
    user_input = st.chat_input("Describe your issue ...")

    if user_input:
        if ml_model is None:
            st.error("ML model not found. Run `python train_model.py` first.")
        else:
            with st.spinner("Analysing and generating response ..."):
                # Sentiment
                sent_result  = predict_sentiment(user_input, ml_model, ml_vectorizer)
                # RAG
                rag_ctx      = rag_engine.get_context_string(user_input, top_k=2)
                # LLM
                reply        = generate_llm_response(user_input, sent_result["label"], rag_ctx)

            st.session_state.chat_history.append({
                "user":      user_input,
                "sentiment": sent_result,
                "reply":     reply,
            })
            st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
