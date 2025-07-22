
import streamlit as st
from llm_backend import answer_question, load_available_languages

st.set_page_config(page_title="Multilingual LLM App", layout="centered")
st.title("🧠 Multilingual Q&A with Local Knowledge Base")

question = st.text_input("Enter your question:")
language = st.selectbox("Select your language:", load_available_languages())

if st.button("Ask"):
    if question.strip():
        with st.spinner("Generating answer..."):
            answer = answer_question(question, language)
            st.success(f"Answer: {answer}")
    else:
        st.warning("Please enter a question.")
