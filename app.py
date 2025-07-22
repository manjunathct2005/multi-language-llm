import streamlit as st
from llm_backend import process_input

# Page config
st.set_page_config(page_title="📚 Smart Multilingual Q&A", layout="centered")

# Title and instructions
st.title("📚 Smart Multilingual Q&A")
st.markdown("### 🤖 Multilingual Knowledge Assistant")
st.markdown("""
Ask questions based on your notes in the following languages:
- **English**
- **Hindi**
- **Telugu**
- **Kannada**
""")

# User input
user_input = st.text_input("💬 Enter your question here:", max_chars=500)

# Process input
if user_input:
    with st.spinner("🔍 Searching your knowledge base..."):
        response = process_input(user_input)
    st.markdown("---")
    st.markdown("### 🧠 Answer")
    st.code(response, language="markdown")
