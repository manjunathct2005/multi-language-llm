import os
import re
import faiss
import torch
import numpy as np
from langdetect import detect
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer

# === CONFIG ===
TEXT_FOLDER = "my1"  # Folder with .txt transcripts
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # For MiniLM-L6-v2

# === Load model and data ===
model = SentenceTransformer(MODEL_NAME)
index = faiss.IndexFlatL2(EMBEDDING_DIM)
sentence_map = []  # List of (sentence, source_file)

def clean_text(text):
    """Remove duplicates, timestamps, and garbage lines."""
    lines = text.split("\n")
    cleaned = []
    seen = set()
    for line in lines:
        line = line.strip()
        if not line or len(line) < 5:
            continue
        line = re.sub(r"\[\d{2}:\d{2}:\d{2}\]", "", line)  # Remove timestamps
        if line not in seen:
            cleaned.append(line)
            seen.add(line)
    return cleaned

def load_documents():
    """Load and embed all text files from folder."""
    for filename in os.listdir(TEXT_FOLDER):
        if filename.endswith(".txt"):
            path = os.path.join(TEXT_FOLDER, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            sentences = clean_text(text)
            embeddings = model.encode(sentences, show_progress_bar=False)
            index.add(np.array(embeddings).astype('float32'))
            sentence_map.extend([(s, filename) for s in sentences])

# Preload on startup
load_documents()

# === Language Utilities ===
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def to_english(text, src_lang):
    if src_lang == "en":
        return text
    return GoogleTranslator(source=src_lang, target="en").translate(text)

def from_english(text, target_lang):
    if target_lang == "en":
        return text
    return GoogleTranslator(source="en", target=target_lang).translate(text)

def load_available_languages():
    """Return list of supported languages by name."""
    return ["en", "hi", "te", "kn"]

# === Main Answering Function ===
def answer_question(question, top_k=1):
    orig_lang = detect_language(question)
    english_q = to_english(question, orig_lang)

    q_emb = model.encode([english_q])[0].astype('float32')
    D, I = index.search(np.array([q_emb]), top_k)

    if I[0][0] == -1:
        return from_english("Sorry, I couldn't find an answer.", orig_lang), None

    matched_sentence, source_file = sentence_map[I[0][0]]
    final_answer = from_english(matched_sentence, orig_lang)
    return final_answer, source_file
