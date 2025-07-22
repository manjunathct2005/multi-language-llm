import os
import warnings
import streamlit as st

# === Suppress Warnings ===
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# === Import Backend ===
try:
    import llm_backend
except ImportError:
    st.error("❌ Could not import `llm_backend.py`. Please ensure it's in the same folder.")
    st.stop()

# === Page Configuration ===
st.set_page_config(page_title="🌍 Multilingual LLM App", layout="centered")
st.title("🌐 Multilingual Question Answering")
st.markdown("Ask a question in **English, Hindi, Telugu, or Kannada**")

# === Input Section ===
query = st.text_input("📝 Enter your question")

# === Process & Display Answer ===
if query:
    with st.spinner("🔍 Searching for answer..."):
        answer = llm_backend.answer_question(query)
        st.success(f"💬 Answer:\n\n{answer}")
