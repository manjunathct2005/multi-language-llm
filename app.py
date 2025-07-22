# app.py
import streamlit as st
from llm_backend import get_answer

st.set_page_config(page_title="🌐 Multilingual Question Answering", layout="centered")
st.title("🌐 Multilingual Question Answering")
st.markdown("Ask a question in **English, Hindi, Telugu, or Kannada**")

question = st.text_input("📝 Enter your question", "")

if question:
    with st.spinner("Thinking..."):
        answer = get_answer(question)
    st.markdown("<br><span style='color:red'><b>Answer:</b></span><br>---<br>" + answer, unsafe_allow_html=True)
