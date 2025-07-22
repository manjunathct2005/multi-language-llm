import streamlit as st
from llm_backend import load_transcripts, get_answer

st.set_page_config(page_title="Multilingual Q&A App", layout="wide")

st.title("🌍 Multilingual Knowledge Assistant")
st.markdown("Ask questions in English, Hindi, Telugu, or Kannada. The system will answer in your language.")

# Load data
with st.spinner("🔄 Loading knowledge base..."):
    texts, embeddings = load_transcripts()

# User Input
query = st.text_input("❓ Ask your question:", placeholder="Type here in your language")

if st.button("🔍 Get Answer") and query.strip():
    with st.spinner("🧠 Generating answer..."):
        try:
            response = get_answer(query, texts, embeddings)
            st.success("✅ Answer:")
            st.write(response)
        except Exception as e:
            st.error("❌ Failed to generate answer.")
            st.exception(e)
