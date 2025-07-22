import streamlit as st
from llm_backend import answer_question

st.set_page_config(page_title="Multilingual Q&A", layout="centered")

st.title("💬 Multilingual Q&A App")
st.markdown("Ask your question in any language (English, Hindi, Telugu, etc.)")

question = st.text_input("Enter your question below:")

if st.button("Get Answer") and question.strip():
    with st.spinner("Thinking..."):
        try:
            answer = answer_question(question)
            st.success("Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"❌ Error: {e}")
