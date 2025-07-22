# app.py

import os
import streamlit as st
import warnings
import torch
import faiss
from llm_backend import (
    transcribe_and_process_audio,
    load_knowledge_base,
    answer_question,
    load_model,
    translate_to_english,
    translate_from_english,
)

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Streamlit page config
st.set_page_config(page_title="Multilingual LLM Tool", layout="wide")
st.title("🌍 Multilingual Q&A from Audio/Text Files")

# Load sentence transformer model using cache_resource
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# Load KB (do not pass model as parameter to avoid UnhashableParamError)
@st.cache_resource
def get_kb():
    return load_knowledge_base()

texts, index, embeddings = get_kb()

# Sidebar for input mode
mode = st.sidebar.radio("Select Input Type", ["Text Input", "Audio File"])

# === TEXT MODE ===
if mode == "Text Input":
    user_input = st.text_area("Enter your question (any supported language):", height=150)
    if st.button("Get Answer"):
        if user_input.strip() == "":
            st.warning("Please enter a question.")
        else:
            # Detect language and translate
            lang = translate_to_english(user_input, detect_only=True)
            translated_q = translate_to_english(user_input)

            # Get answer from knowledge base
            answer = answer_question(translated_q, texts, index, embeddings, model)

            # Translate back to user language
            final_answer = translate_from_english(answer, lang)
            st.success(f"**Answer ({lang}):**\n{final_answer}")

# === AUDIO MODE ===
elif mode == "Audio File":
    uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a"])
    if uploaded_file is not None:
        with st.spinner("Transcribing and processing..."):
            question = transcribe_and_process_audio(uploaded_file)

        if question:
            st.info(f"**Transcribed Question:** {question}")
            lang = translate_to_english(question, detect_only=True)
            translated_q = translate_to_english(question)

            answer = answer_question(translated_q, texts, index, embeddings, model)
            final_answer = translate_from_english(answer, lang)
            st.success(f"**Answer ({lang}):**\n{final_answer}")
        else:
            st.error("Could not transcribe the audio file.")
