import streamlit as st
from llm_backend import load_transcripts, answer_question

st.set_page_config(page_title="Multilingual QA App", layout="centered")
st.title("🧠 Multilingual Question Answering")
st.markdown("Ask in **Hindi**, **Telugu**, **Kannada**, or **English**")

with st.spinner("Loading knowledge base..."):
    texts, embeddings = load_transcripts()

query = st.text_input("💬 Your question:")

if query:
    with st.spinner("Analyzing..."):
        try:
            answer = answer_question(query, texts, embeddings)
            st.success("✅ Answer:")
            st.markdown(answer)
        except Exception as e:
            st.error("⚠️ Sorry! Could not process your request.")
            st.exception(e)
