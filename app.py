# app.py

import os
import streamlit as st
from llm_backend import process_input

st.set_page_config(page_title="🌐 Multilingual LLM App", layout="centered")
st.title("💬 Multilingual Q&A (Telugu, Hindi, English)")

user_input = st.text_input("Ask your question in Telugu / Hindi / English:")

if user_input:
    with st.spinner("Processing..."):
        response = process_input(user_input)
    st.success("Answer:")
    st.write(response)
