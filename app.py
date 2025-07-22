import streamlit as st
from llm_backend import search_answer

st.set_page_config(page_title="Multilingual LLM Tool", layout="centered")
st.title("🧠 Multilingual LLM - Text Knowledge Base")

query = st.text_input("Ask your question (any language):")

if query:
    with st.spinner("Searching..."):
        response = search_answer(query)
        st.success("Answer:")
        st.markdown(response)
