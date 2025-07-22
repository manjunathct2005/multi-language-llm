import streamlit as st
from llm_backend import answer_question, load_available_languages

st.set_page_config(page_title="Multilingual Q&A", layout="centered")
st.title("🌍 Multilingual Q&A Tool")

st.markdown("Ask your question in any supported language (e.g., English, Hindi, Telugu, Kannada) and get the answer in the same language.")

question = st.text_input("🔍 Enter your question:")

if question:
    with st.spinner("Processing..."):
        answer, source = answer_question(question)

        if answer:
            st.markdown(f"### ✅ Answer:\n<span style='color:red'>{answer}</span>", unsafe_allow_html=True)
            if source:
                st.markdown(f"**📁 Source:** `{source}`")
        else:
            st.warning("Sorry, I couldn't find a relevant answer.")
