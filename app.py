# 📘 app.py - Streamlit interface for the Multilingual Knowledge Assistant

import streamlit as st
from llm_backend import process_input

# Page Configuration
st.set_page_config(
    page_title="📚 Smart Multilingual Q&A",
    layout="centered"
)

# App Title
st.markdown("## 🤖 Multilingual Knowledge Assistant")

# App Description
st.markdown("""
Ask questions about your notes in:
- **English**
- **Hindi**
- **Telugu**
- **Kannada**

This tool includes:
- Relevant code snippets (if applicable)
- Precise, language-matched answers
- Fast response from your local knowledge base
""")

# User Input Section
user_input = st.text_input("💬 Enter your question:", max_chars=500)

# Output Response Section
if user_input:
    with st.spinner("🔍 Thinking..."):
        response = process_input(user_input)
        st.markdown("---")
        st.markdown("### 🧠 Answer:")
        st.markdown(response, unsafe_allow_html=True)
