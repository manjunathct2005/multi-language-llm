import os
import streamlit as st

# Check local module import
try:
    from llm_backend import answer_question, load_available_languages
except ImportError:
    st.error("❌ Failed to import llm_backend.py. Make sure it's in the same folder as app.py.")
    st.stop()

st.set_page_config(page_title="Multilingual Q&A Tool", layout="centered")
st.title("🌍 Multilingual Question Answering App")

# Sidebar
st.sidebar.header("Language Settings")
supported_langs = load_available_languages()
st.sidebar.markdown("Available languages: " + ", ".join(supported_langs))

# Input
query = st.text_input("Ask your question (in any language):")

if query:
    with st.spinner("Searching for answer..."):
        try:
            answer, source = answer_question(query)
            st.success("✅ Answer:")
            st.write(answer)
            st.caption(f"📁 Source file: {source}")
        except Exception as e:
            st.error(f"⚠️ Something went wrong: {str(e)}")

st.markdown("---")
st.caption("💡 English, Hindi, Telugu, Kannada supported. Local model, no API required.")
