import streamlit as st
from llm_backend import answer_question, texts

st.set_page_config(page_title="Multilingual LLM QA", layout="centered")
st.title("🧠 Multilingual Q&A Chatbot")

user_input = st.text_input("Ask your question:")

if user_input:
    answer = answer_question(user_input.strip(), texts)
    st.markdown("### ✅ Answer:")
    st.write(answer)
