import warnings
import os
import streamlit as st
from llm_backend import answer_question

# Suppress warnings and logs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Streamlit UI setup
st.set_page_config(page_title="Multilingual LLM Tool", layout="centered")
st.title("💬 Multilingual Q&A App")
st.markdown("Ask a question in **English, Hindi, Telugu, or Kannada**.")

# Input
question = st.text_input("Enter your question here:")

# Process
if question.strip():
    with st.spinner("🔍 Searching..."):
        try:
            answer, lang, _ = answer_question(question)
            st.markdown(f"<span style='color:green;font-weight:bold;'>🧠 Answer ({lang.upper()}):</span>", unsafe_allow_html=True)
            st.success(answer)
        except Exception as e:
            st.markdown("<span style='color:red;font-weight:bold;'>❌ Error occurred:</span>", unsafe_allow_html=True)
            st.error(str(e))
