import os
import re
import faiss
import torch
import numpy as np
from langdetect import detect
from deep_translator import MyMemoryTranslator
from sentence_transformers import SentenceTransformer

# === CONFIG ===
TEXT_FOLDER = r"D:\llm project\my1"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# === Load Model ===
model = SentenceTransformer(EMBEDDING_MODEL)

# === Translation Helpers ===
def translate_to_english(text):
    try:
        return MyMemoryTranslator(source='auto', target='en').translate(text)
    except:
        return text

def translate_from_english(text, target_lang):
    try:
        return MyMemoryTranslator(source='en', target=target_lang).translate(text)
    except:
        return text

# === Cleaning ===
def clean_text(text):
    text = re.sub(r"\n+", ". ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\.{2,}", ".", text)
    return text.strip()

# === Chunking ===
def chunk_text(text, max_length=500):
    words = text.split()
    return [' '.join(words[i:i+max_length]) for i in range(0, len(words), max_length)]

# === Load & Embed Files ===
def load_chunks_and_embeddings():
    all_chunks, sources = [], []
    for filename in os.listdir(TEXT_FOLDER):
        if filename.endswith(".txt"):
            filepath = os.path.join(TEXT_FOLDER, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = clean_text(f.read())
                chunks = chunk_text(text)
                all_chunks.extend(chunks)
                sources.extend([filename] * len(chunks))
    if not all_chunks:
        return [], [], None
    embeddings = model.encode(all_chunks, convert_to_tensor=False, normalize_embeddings=True)
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))
    return all_chunks, sources, index

CHUNKS, SOURCES, INDEX = load_chunks_and_embeddings()

# === Q&A ===
def answer_question(query):
    if not CHUNKS or not INDEX:
        return "⚠️ No knowledge base found. Please upload .txt files."

    original_lang = detect(query)
    query_en = translate_to_english(query)
    q_embed = model.encode([query_en], convert_to_tensor=False, normalize_embeddings=True)
    top_k = 3
    _, indices = INDEX.search(np.array(q_embed), top_k)

    matched_chunks = [CHUNKS[i] for i in indices[0]]
    answer_en = "\n\n".join(matched_chunks)

    if not answer_en.strip():
        return translate_from_english("Sorry, I could not find an answer.", original_lang)

    return translate_from_english(answer_en, original_lang)
