import streamlit as st
from llm_backend import answer_question, load_available_languages

st.set_page_config(page_title="Multilingual LLM QA", layout="centered")
st.title("🧠 Multilingual Local Q&A")

st.markdown("Ask questions in **English** or **Telugu**. The system will find answers from your local `.txt` files.")

question = st.text_input("Your Question:")
language = st.selectbox("Language:", load_available_languages())

if st.button("Ask"):
    if question.strip():
        with st.spinner("Searching for the answer..."):
            response = answer_question(question)
            st.success(response)
    else:
        st.warning("Please enter a valid question.")
