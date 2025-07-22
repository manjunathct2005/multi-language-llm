import streamlit as st
from llm_backend import load_transcripts, answer_question

st.set_page_config(page_title="Multilingual QA", layout="centered")
st.title("🌐 Multilingual Q&A Chatbot")
st.markdown("Ask in **English**, **Hindi**, **Telugu**, or **Kannada**")

with st.spinner("📚 Loading knowledge base..."):
    texts, embeddings = load_transcripts()

user_input = st.text_input("🗨️ Ask your question here:")

if user_input:
    with st.spinner("🔍 Searching..."):
        try:
            response = answer_question(user_input, texts, embeddings)
            st.success("✅ Answer:")
            st.write(response)
        except Exception as e:
            st.error("❌ Failed to generate an answer.")
            st.text(str(e))
