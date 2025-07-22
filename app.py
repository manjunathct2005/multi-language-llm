import streamlit as st
from llm_backend import get_answer

st.set_page_config(page_title="Multilingual LLM Q&A", layout="centered")
st.title("🌍 Multilingual LLM QA Tool")

st.write("Ask a question in **Hindi, Telugu, Kannada, or English**.")

query = st.text_input("Enter your question:", "")

if query:
    with st.spinner("Generating answer..."):
        answer = get_answer(query)
    st.success("✅ Answer:")
    st.write(answer)
