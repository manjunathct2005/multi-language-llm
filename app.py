import os
import streamlit as st
from llm_backend import answer_question, load_available_languages

# === PAGE CONFIG ===
st.set_page_config(page_title="Multilingual LLM Q&A", layout="centered")
st.title("🌐 Multilingual LLM Q&A App")
st.markdown("Ask any question based on your uploaded transcripts!")

# === USER INPUT ===
query = st.text_input("💬 Enter your question here")

if st.button("Ask"):
    if query.strip():
        with st.spinner("Processing..."):
            try:
                response = answer_question(query)
                st.success("✅ Answer:")
                st.write(response)
            except Exception as e:
                st.error("⚠️ Something went wrong.")
                st.exception(e)
    else:
        st.warning("❗ Please enter a question first.")

# === FOOTER ===
st.markdown("---")
st.markdown("Built with 💻 by Manju Nath using local models only")
