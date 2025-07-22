# app.py

import os
import streamlit as st
from llm_backend import process_input, knowledge_base, detect_language

st.set_page_config(page_title="Multilingual Knowledge Base Assistant", layout="centered")

# Initialize
TRANSCRIPT_DIR = "D:/hindupur_dataset/transcripts"
SUPPORTED_LANGUAGES = ["en", "hi", "te"]

# Load knowledge base
texts, index, embeddings = knowledge_base(TRANSCRIPT_DIR)
num_blocks = len(texts)

# Header
st.markdown("## 💡 Multilingual Knowledge Base Assistant")
st.markdown("Ask in **Telugu**, **Hindi**, **Kannada**, or **English**. Clean answers from `.txt` transcripts.")

# Knowledge base status
if num_blocks > 0:
    st.success(f"✅ Loaded {num_blocks} knowledge blocks from `{TRANSCRIPT_DIR}`.")
else:
    st.warning("⚠️ No knowledge blocks found. Please add `.txt` transcripts.")

# Input box
query = st.text_input("🔎 Ask your question here:")

# Response style
st.markdown("#### 📝 Response Style:")
response_style = st.radio("", ["Summary", "Detailed (Chat-style)"], horizontal=True)

# Submit button
if st.button("🚀 Get Answer"):

    if not query.strip():
        st.error("❌ Please enter a valid question.")
    else:
        detected_lang = detect_language(query)

        if detected_lang not in SUPPORTED_LANGUAGES:
            st.error("❌ Only Telugu/English/Hindi questions are supported.")
        else:
            result = process_input(query, texts, index, embeddings, detected_lang, style=response_style)
            st.markdown("### ✅ Answer:")
            st.write(result)
