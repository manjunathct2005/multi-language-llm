import os
import streamlit as st
from llm_backend import answer_question, load_available_languages

# Streamlit config
st.set_page_config(page_title="Multilingual Q&A Tool", layout="centered")
st.title("🌍 Multilingual Question Answering App")

# Sidebar for language info
st.sidebar.header("🔤 Supported Languages")
supported_langs = load_available_languages()
st.sidebar.markdown("You can ask in: " + ", ".join(supported_langs))

# Main input box
query = st.text_input("❓ Ask your question (in any supported language):")

# Process query
if query:
    with st.spinner("🔍 Searching for the answer..."):
        try:
            answer, source = answer_question(query)
            st.success("✅ Answer:")
            st.write(answer)
            st.caption(f"📂 Source file: {source}")
        except Exception as e:
            st.error("⚠️ An error occurred while answering the question.")
            st.exception(e)  # Optional: shows full traceback for debugging

# Footer
st.markdown("---")
st.caption("🔗 This tool supports multilingual Q&A using Sentence Transformers and Google Translator locally.")
