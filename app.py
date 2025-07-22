# app.py

import streamlit as st
from llm_backend import get_answer

st.set_page_config(page_title="🌐 Multilingual LLM QA", layout="centered")
st.title("🌐 Multilingual Question Answering")
st.markdown("Ask a question in **English, Hindi, Telugu, or Kannada**")

# Input box
question = st.text_input("📝 Enter your question", "")

# Get Answer
if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = get_answer(question)
        st.markdown(f"<span style='color:red'><b>Answer:</b></span><br>{answer}", unsafe_allow_html=True)
