import streamlit as st
from llm_backend import get_answer

st.set_page_config(page_title="Multilingual Q&A Chatbot", layout="centered")
st.markdown("## 🌐 Multilingual Q&A Chatbot")
st.markdown("Ask in **English**, **Hindi**, **Telugu**, or **Kannada**")

query = st.text_input("💬 Ask your question here:", "")

if st.button("🔍 Get Answer"):
    if query.strip():
        with st.spinner("Generating answer..."):
            try:
                answer = get_answer(query)
                st.success(f"🗣️ Answer: {answer}")
            except Exception as e:
                st.error(f"❌ Failed to generate an answer.\n\n{str(e)}")
    else:
        st.warning("Please enter a question to continue.")
