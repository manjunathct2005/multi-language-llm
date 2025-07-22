# ================= llm_backend.py =================
import os
import torch
import re
import numpy as np
import faiss
from langdetect import detect
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util

# === CONFIG ===
TEXT_FOLDER = r"D:\llm project\my1"
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# === Load model ===
model = SentenceTransformer(MODEL_NAME)

# === Preprocess and load text ===
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_transcripts():
    chunks, sources = [], []
    for file in os.listdir(TEXT_FOLDER):
        if file.endswith(".txt"):
            with open(os.path.join(TEXT_FOLDER, file), "r", encoding="utf-8") as f:
                content = f.read().split("\n")
                for line in content:
                    line = clean_text(line)
                    if 20 < len(line) < 1000:
                        chunks.append(line)
                        sources.append(file)
    return chunks, sources

CHUNKS, SOURCES = load_transcripts()
EMBEDDINGS = model.encode(CHUNKS, convert_to_tensor=True)

# === Language utils ===
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def translate_to_english(text):
    lang = detect_language(text)
    if lang != "en":
        return GoogleTranslator(source=lang, target="en").translate(text), lang
    return text, "en"

def translate_from_english(text, target_lang):
    if target_lang != "en":
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    return text

# === Q&A ===
def answer_question(query):
    translated_query, orig_lang = translate_to_english(query)
    query_embedding = model.encode(translated_query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, EMBEDDINGS)[0]
    top_idx = torch.argmax(scores).item()
    answer = CHUNKS[top_idx]
    translated_answer = translate_from_english(answer, orig_lang)
    return translated_answer, SOURCES[top_idx]

def load_available_languages():
    return ["en", "hi", "te", "kn"]  # You can extend this


# ================= app.py =================
import os
import streamlit as st
from llm_backend import answer_question, load_available_languages

st.set_page_config(page_title="Multilingual Q&A Tool", layout="centered")
st.title("🌍 Multilingual Question Answering App")

# Sidebar options
st.sidebar.header("Language Settings")
supported_langs = load_available_languages()
st.sidebar.markdown("Available languages: " + ", ".join(supported_langs))

# Main query input
query = st.text_input("Ask your question (Any language supported):")

if query:
    with st.spinner("Generating answer..."):
        try:
            answer, source = answer_question(query)
            st.success("Answer:")
            st.write(answer)
            st.caption(f"📁 Source file: {source}")
        except Exception as e:
            st.error(f"Something went wrong. Details: {e}")

st.markdown("---")
st.caption("💡 Supports English, Hindi, Telugu, Kannada. Uses local embeddings and multilingual translation.")
