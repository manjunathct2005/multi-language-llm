# app.py

import streamlit as st
from llm_backend import get_answer

# App title and layout
st.set_page_config(page_title="🌐 Multilingual Question Answering", layout="centered")

st.title("🌐 Multilingual Question Answering")
st.markdown("Ask a question in **English**, **Hindi**, **Telugu**, or **Kannada**")

# Input
query = st.text_input("📝 Enter your question", placeholder="e.g., What is Data Science?")

# Answer Button
if st.button("💬 Answer"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching for best answer..."):
            response = get_answer(query)
        st.markdown(f"<span style='color:red'><b>Answer:</b></span><br>{response}", unsafe_allow_html=True)
