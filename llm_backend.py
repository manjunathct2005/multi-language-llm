import os
import re
import faiss
import torch
import numpy as np
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
from googletrans import Translator

# ========== CONFIG ==========
TEXT_FOLDER = r"D:\llm project\my1"  # Adjust path as needed
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ========== LOAD EMBEDDING MODEL ==========
model = SentenceTransformer(MODEL_NAME)

# ========== LANGUAGE DETECTION ==========
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# ========== TRANSLATION ==========
translator = Translator()

def translate_to_english(text):
    return translator.translate(text, dest='en').text

def translate_from_english(text, target_lang):
    return translator.translate(text, dest=target_lang).text

# ========== LOAD TRANSCRIPTS AND CREATE INDEX ==========
def clean_text(text):
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

@st.cache_data(show_spinner=False)
def load_knowledge_base():
    texts = []
    for file in os.listdir(TEXT_FOLDER):
        if file.endswith(".txt"):
            with open(os.path.join(TEXT_FOLDER, file), "r", encoding="utf-8") as f:
                content = f.read()
                content = clean_text(content)
                if content:
                    texts.append(content)

    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.cpu().detach().numpy())

    return texts, index, embeddings

# ========== QUERY ANSWER ==========
def get_answer(question, texts, index, embeddings):
    lang = detect_language(question)
    translated_q = translate_to_english(question) if lang != "en" else question

    q_embedding = model.encode(translated_q, convert_to_tensor=True)
    D, I = index.search(q_embedding.cpu().numpy().reshape(1, -1), 1)

    answer = texts[I[0][0]] if I[0][0] < len(texts) else "Sorry, I couldn’t find the answer."
    return translate_from_english(answer, lang) if lang != "en" else answer

# ========== MAIN INTERFACE ==========
if __name__ == "__main__":
    import streamlit as st
    st.set_page_config(page_title="Multilingual QA", layout="centered")

    st.title("🌐 Multilingual Q&A using Local Embeddings")
    user_input = st.text_area("Ask your question:", "", height=100)

    if user_input:
        with st.spinner("Loading knowledge base..."):
            texts, index, embeddings = load_knowledge_base()

        with st.spinner("Searching for the answer..."):
            answer = get_answer(user_input, texts, index, embeddings)

        st.success("Answer:")
        st.write(answer)
