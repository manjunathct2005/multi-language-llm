import streamlit as st
from llm_backend import answer_question, load_available_languages

st.set_page_config(page_title="🧠 Multilingual LLM Q&A", layout="centered")
st.title("🌐 Multilingual Question Answering")
st.write("Ask a question in **English or Telugu**. The app will search local knowledge files.")

question = st.text_input("📝 Enter your question:")
language = st.selectbox("🌐 Select your language:", load_available_languages())

if st.button("Ask"):
    if question.strip():
        with st.spinner("Thinking..."):
            answer = answer_question(question, language.lower()[0:2])
            st.success("💬 Answer:")
            st.write(answer)
    else:
        st.warning("Please enter a question to get an answer.")
