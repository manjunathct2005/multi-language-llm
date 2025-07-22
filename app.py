import os
import streamlit as st
from llm_backend import search_answer

st.set_page_config(page_title="Multilingual LLM", layout="centered")
st.title("🌍 Multilingual LLM Q&A App")
st.markdown("Ask a question in **Hindi**, **Telugu**, **Kannada**, or **English**.")

user_input = st.text_input("💬 Enter your question:")

if st.button("Get Answer"):
    if user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching..."):
            answer = search_answer(user_input)
            st.success("✅ Answer:")
            st.markdown(f"**{answer}**")
