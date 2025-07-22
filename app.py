import streamlit as st
from llm_backend import search_answer

st.set_page_config(page_title="Multilingual Q&A App", layout="centered")
st.title("🌍 Multilingual LLM Assistant")
st.markdown("Ask any question in **Hindi, Telugu, Kannada, or English**.")

# Input
user_input = st.text_area("🔤 Enter your question:", height=100)

if st.button("Get Answer"):
    if user_input.strip() == "":
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Thinking..."):
            response = search_answer(user_input)
            st.success("✅ Answer:")
            st.write(response)
