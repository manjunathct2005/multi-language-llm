import os
import re
import faiss
import torch
import numpy as np
from langdetect import detect
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer

# === CONFIG ===
TEXT_FOLDER = r"D:\llm project\my1"  # Change to your folder path
MODEL_NAME = "all-MiniLM-L6-v2"

# === LOAD MODEL ===
model = SentenceTransformer(MODEL_NAME)

# === HELPER FUNCTIONS ===

def translate(text, src, tgt):
    if src == tgt:
        return text
    return GoogleTranslator(source=src, target=tgt).translate(text[:5000])

def clean_text(text):
    lines = text.splitlines()
    seen = set()
    cleaned = []
    for line in lines:
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            cleaned.append(line)
    return cleaned

def load_transcripts(text_folder):
    texts, sources = [], []
    for filename in os.listdir(text_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(text_folder, filename), "r", encoding="utf-8") as f:
                content = f.read()
                for line in clean_text(content):
                    texts.append(line)
                    sources.append(filename)
    return texts, sources

def build_embeddings(texts):
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=16)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

# === LOAD KNOWLEDGE BASE ===

texts, sources = load_transcripts(TEXT_FOLDER)
index, _ = build_embeddings(texts)

# === ANSWER ENGINE ===

def search_answer(query):
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb), k=5)
    results = [texts[i] for i in I[0]]
    return results

def format_bullet_answer(results):
    bullets = "\n".join(f"- {r}" for r in results if r.strip())
    return bullets.strip() if bullets else None

def process_input(user_input):
    lang = detect(user_input)
    query_en = translate(user_input, lang, "en")
    results = search_answer(query_en)
    if results:
        bullets = format_bullet_answer(results)
        answer_en = bullets if bullets else results[0]
        return translate(answer_en, "en", lang)
    return translate("Sorry, I couldn't find the answer in the knowledge base.", "en", lang)
