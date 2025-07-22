# app.py

import streamlit as st
import os
from llm_backend import load_texts_and_embeddings, process_input

# Set up Streamlit config
st.set_page_config(page_title="Multilingual QA Tool", layout="centered")
st.title("🌐 Multilingual Question Answering App")
st.markdown("Ask a question in **English, Hindi, Telugu, or any supported language**. The system will reply in the same language.")

# Load data
@st.cache_resource
def load_resources():
    texts, index, embeddings = load_texts_and_embeddings()
    return texts, index

with st.spinner("🔄 Loading knowledge base..."):
    texts, index = load_resources()

# Input UI
user_input = st.text_input("📝 Enter your question:", placeholder="Type here...")
submit = st.button("🔍 Get Answer")

if submit and user_input.strip():
    with st.spinner("🧠 Thinking..."):
        try:
            answer = process_input(user_input, texts, index)
            st.success("💡 Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"❌ Error while answering: {str(e)}")

# Optional: show debug logs or metadata
with st.expander("⚙️ Debug Info"):
    st.write("📁 Text files loaded:", len(texts))
    st.write("✅ Embedding index ready.")
