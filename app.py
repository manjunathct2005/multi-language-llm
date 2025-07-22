# app.py

import streamlit as st
from llm_backend import search_answer

st.set_page_config(page_title="Multilingual Q&A App", layout="centered")
st.title("🧠 Multilingual Q&A from My Text Files")

with st.form("query_form"):
    user_query = st.text_area("Ask a question in any language:", height=150)
    submitted = st.form_submit_button("🔍 Get Answer")

if submitted and user_query.strip():
    with st.spinner("Searching the best answer..."):
        try:
            response = search_answer(user_query)
            st.success("✅ Answer:")
            st.markdown(response)
        except Exception as e:
            st.error(f"❌ Error occurred: {str(e)}")
else:
    st.info("Enter your question and press the button to get started.")
