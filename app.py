import os
import streamlit as st
from llm_backend import answer_question

st.set_page_config(page_title="Multilingual LLM Q&A", layout="centered")
st.title("🌍 Multilingual Q&A LLM")
st.write("Ask your question in Hindi, Telugu, Kannada, or English.")

query = st.text_input("💬 Enter your question:")

if query:
    with st.spinner("🔍 Finding answer..."):
        answer, source = answer_question(query)
        st.success("✅ Answer:")
        st.markdown(f"**{answer}**")
        st.caption(f"📁 Source: {source}")
