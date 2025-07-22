import os
import streamlit as st

# UI Configuration
st.set_page_config(page_title="Multilingual Q&A Tool", layout="centered")
st.title("🌍 Multilingual Question Answering App")

# Import backend
try:
    from llm_backend import answer_question, load_available_languages
except ImportError:
    st.error("❌ Could not import `llm_backend.py`. Please ensure it's in the same directory.")
    st.stop()

# Sidebar Language Info
st.sidebar.header("Language Settings")
try:
    supported_langs = load_available_languages()
    st.sidebar.markdown("✅ Supported: " + ", ".join(supported_langs))
except Exception as e:
    st.sidebar.error("❌ Could not load languages.")
    supported_langs = []

# Question Input
query = st.text_input("❓ Ask your question (in English, Hindi, Telugu, Kannada):")

if query:
    with st.spinner("🔍 Finding the best answer..."):
        try:
            answer, source = answer_question(query)
            if answer:
                st.success("✅ Answer:")
                st.markdown(f"**{answer}**")
                if source:
                    st.caption(f"📁 From: `{source}`")
            else:
                st.warning("⚠️ No relevant answer found.")
        except Exception as e:
            st.error(f"❌ Error during answering: {e}")

st.markdown("---")
st.caption("🤖 Powered by local models. No internet/API required.")
