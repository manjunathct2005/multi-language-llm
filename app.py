import streamlit as st
from llm_backend import process_input

st.set_page_config(page_title="📚 Smart Multilingual Q&A", layout="centered")
st.title("🤖 Multilingual Knowledge Assistant")
st.write("Ask anything from your custom data in **English, Hindi, Telugu, or Kannada**.")

user_input = st.text_input("💬 Ask a question", max_chars=500, placeholder="e.g. इस विषय के बारे में बताएं...")

if user_input:
    with st.spinner("🔍 Thinking..."):
        answer = process_input(user_input)
        st.markdown("---")
        st.markdown(answer, unsafe_allow_html=True)
