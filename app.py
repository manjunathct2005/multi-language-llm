# app.py
import streamlit as st
from llm_backend import get_answer

st.set_page_config(page_title="🌐 Multilingual LLM", layout="centered")
st.title("🌐 Multilingual Question Answering")
st.markdown("Ask a question in **English**, **Hindi**, **Telugu**, or **Kannada**.")

# Input box
query = st.text_input("📝 Enter your question", max_chars=500)

# Process
if st.button("💬 Get Answer"):
    if query.strip() == "":
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Thinking..."):
            answer = get_answer(query)
            st.markdown("<span style='color:red'><b>Answer:</b></span><br>---<br>" + answer, unsafe_allow_html=True)
