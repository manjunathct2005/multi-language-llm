import os
import warnings
import streamlit as st
from llm_backend import (
    load_knowledge_base,
    detect_language,
    translate_question,
    retrieve_answer,
)

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

st.set_page_config(page_title="Multilingual LLM Tool", layout="centered")

st.title("🧠 Multilingual Q&A App")
st.markdown("Ask questions in **Telugu, Hindi, Kannada, or English**. Get answers from transcripts!")

# === Load Model and Knowledge Base (no caching for unhashable types) ===
with st.spinner("🔄 Loading model and knowledge base..."):
    model, texts, index, embeddings = load_knowledge_base()

# === Input Section ===
query = st.text_input("🔍 Enter your question:")

if query:
    with st.spinner("💬 Processing your question..."):
        detected_lang = detect_language(query)
        translated_q = translate_question(query, detected_lang)

        answer = retrieve_answer(translated_q, model, texts, index, embeddings)

        if answer:
            st.success("✅ Answer:")
            st.write(answer if detected_lang == "en" else translate_question(answer, "en", detected_lang))
        else:
            st.warning("⚠️ No relevant answer found in the knowledge base.")
            st.markdown(f"Try refining your question or [search on Google](https://www.google.com/search?q={query}) 🔍")
