import streamlit as st
from llm_backend import get_answer

st.set_page_config(page_title="Multilingual LLM QA", layout="centered")
st.title("🌍 Multilingual Q&A App")

question = st.text_input("Enter your question (Hindi, Telugu, Kannada, or English):")

if st.button("Get Answer"):
    if question:
        with st.spinner("Thinking..."):
            answer = get_answer(question)
        st.success("Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a question.")
