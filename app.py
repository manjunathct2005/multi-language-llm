import streamlit as st
from llm_backend import process_input

st.set_page_config(page_title="Multilingual LLM QA", layout="centered")
st.title("🌍 Multilingual Q&A App")

st.markdown("Ask a question in **any language** based on your `.txt` files.")

user_input = st.text_input("🔍 Ask something:")

if user_input:
    with st.spinner("Searching..."):
        response = process_input(user_input)
    st.success(response)
