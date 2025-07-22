import os
import warnings
import streamlit as st
from llm_backend import knowledge_base, process_input, detect_language

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# === Streamlit App Configuration ===
st.set_page_config(page_title="🧠 Multilingual Text Q&A", layout="centered")
st.title("📄 Multilingual Transcript Q&A")
st.markdown("Ask questions from the uploaded `.txt` transcripts (multilingual supported).")

# === Load Knowledge Base ===
@st.cache_resource
def load_kb():
    with st.spinner("Loading knowledge base and embeddings..."):
        return knowledge_base()

try:
    texts, index, embeddings = load_kb()
except Exception as e:
    st.error(f"Failed to load knowledge base: {e}")
    st.stop()

# === User Input Section ===
st.subheader("🔍 Ask your question")
user_query = st.text_input("Enter your question here", placeholder="Type your question...")

mode = st.selectbox("Select Response Mode", ["summary", "detailed"])

# === Answer Generation ===
if st.button("Get Answer") and user_query.strip():
    with st.spinner("Processing your question..."):
        lang = detect_language(user_query)
        answer = process_input(user_query, mode, texts, index, embeddings, lang)

    st.markdown("### ✅ Answer:")
    st.write(answer)
