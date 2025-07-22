import os
import streamlit as st
from llm_backend import load_texts_and_embeddings, process_input

st.set_page_config(page_title="Multilingual LLM Q&A", layout="centered")

st.title("🌐 Multilingual Q&A with Local Knowledge Base")

@st.cache_resource(show_spinner="Loading data...")
def load_resources():
    texts, index = load_texts_and_embeddings()
    return texts, index

texts, index = load_resources()

user_question = st.text_input("Enter your question (in any language):")

if user_question:
    with st.spinner("Processing..."):
        answer = process_input(user_question, texts, index)
        st.markdown("### Answer:")
        st.success(answer)
