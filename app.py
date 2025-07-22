import streamlit as st
from llm_backend import answer_question

st.set_page_config(page_title="📚 Smart Multilingual Q&A", layout="centered")

st.title("🤖 Multilingual Knowledge Assistant")
st.markdown("Ask questions from your notes in **English, Hindi, Telugu, or Kannada.**")

user_input = st.text_input("💬 Enter your question", max_chars=500)

if user_input:
    with st.spinner("🔍 Searching..."):
        response = answer_question(user_input)
        st.markdown("---")
        st.markdown(response, unsafe_allow_html=True)
