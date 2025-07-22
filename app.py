import warnings
import os
import streamlit as st
from llm_backend import answer_question, load_available_languages

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

st.set_page_config(page_title="Multilingual Q&A LLM", layout="centered")

st.title("🌐 Multilingual LLM Q&A Tool")
st.markdown("Ask any question in your language. It will search the knowledge base and respond in the same language.")

# Load available languages from backend
available_langs = load_available_languages()

# Input method
input_type = st.radio("Select Input Type:", ["Text Question", "Upload Audio"], horizontal=True)

# Handle text question
if input_type == "Text Question":
    user_question = st.text_input("❓ Enter your question", max_chars=500)
    if st.button("Get Answer") and user_question.strip() != "":
        with st.spinner("Processing..."):
            answer = answer_question(user_question)
        st.markdown("**Answer:**")
        st.success(answer)

# Handle audio input
else:
    uploaded_audio = st.file_uploader("🔊 Upload Audio File (.mp3, .wav)", type=["mp3", "wav"])
    if uploaded_audio is not None and st.button("Transcribe & Answer"):
        with st.spinner("Transcribing..."):
            answer = answer_question(uploaded_audio)
        st.markdown("**Answer:**")
        st.success(answer)

st.markdown("---")
st.markdown("💬 **Supported Languages**: " + ", ".join(available_langs))
