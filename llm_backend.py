import os
import streamlit as st
from llm_backend import get_answer

st.set_page_config(page_title="Multilingual LLM Q&A", layout="centered")

st.title("🌐 Multilingual Q&A using Local LLM")
st.markdown("Ask any question in your language. The app will respond in the same language.")

# === User Input ===
user_input = st.text_input("📝 Ask a question:")

# === Process ===
if st.button("Get Answer"):
    if user_input.strip() == "":
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Processing..."):
            try:
                response = get_answer(user_input)
                st.success(f"💬 Answer: {response}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
