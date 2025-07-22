import os
import streamlit as st
from llm_backend import answer_question

# Streamlit Config
st.set_page_config(page_title="🌍 Multilingual LLM", layout="centered")
st.title("🌍 Multilingual LLM QA App")
st.markdown("Ask a question in **Hindi, Telugu, Kannada, English** or others. You'll get the answer in the same language!")

# Input Box
query = st.text_area("🗣️ Enter your question below:", height=100)

# Process
if st.button("🔍 Get Answer"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking... 🤔"):
            answer = answer_question(query)
            st.success("✅ Answer:")
            st.write(answer)
