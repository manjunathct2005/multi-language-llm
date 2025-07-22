import os
import streamlit as st
from llm_backend import get_answer

st.set_page_config(page_title="🌐 Multilingual QA", layout="centered")
st.title("🌐 Multilingual Question Answering")
st.markdown("Ask a question in **English**, **Hindi**, **Telugu**, or **Kannada**")

st.markdown("### 📝 Enter your question")

query = st.text_input("", placeholder="e.g., डाटा साइंस क्या है?", key="user_query")

if st.button("💬 Get Answer") and query.strip():
    with st.spinner("Analyzing..."):
        try:
            answer = get_answer(query)
            st.markdown(
                f"<span style='color:red'><b>Answer:</b></span><br>{answer}",
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error("⚠️ Error while answering your question.")
else:
    st.info("Enter a question above and click 'Get Answer'.")
