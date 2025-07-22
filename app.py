import streamlit as st
from llm_backend import answer_question

st.set_page_config(page_title="Multilingual Q&A", layout="centered")
st.markdown("<h2 style='text-align: center;'>🌍 Multilingual Q&A System</h2>", unsafe_allow_html=True)

st.markdown("Type your question in **English**, **Hindi**, **Telugu**, **Kannada**, or other supported languages.")

user_input = st.text_input("💬 Enter your question:")

if st.button("Get Answer"):
    if user_input.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching for the best answer..."):
            answer, lang = answer_question(user_input)
            st.markdown(answer, unsafe_allow_html=True)
