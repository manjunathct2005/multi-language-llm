import os
import re
import faiss
import torch
import numpy as np
from langdetect import detect
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

TEXT_FOLDER = "my1"
MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

# Clean and chunk logic
def clean_and_chunk(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove emojis/non-ASCII
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"---+", "\n---\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    raw_chunks = re.split(r"\n\s*---\s*\n", text)
    clean_chunks = []

    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk or len(chunk) < 50:
            continue
        lines = chunk.split("\n")
        cleaned = "\n".join([line.strip() for line in lines if line.strip()])
        clean_chunks.append(cleaned)

    return clean_chunks

# Load and clean all documents
def load_documents():
    all_chunks = []
    for file in os.listdir(TEXT_FOLDER):
        if file.endswith(".txt"):
            with open(os.path.join(TEXT_FOLDER, file), encoding="utf-8") as f:
                text = f.read()
                chunks = clean_and_chunk(text)
                all_chunks.extend(chunks)
    return all_chunks

print("[✓] Loading documents...")
knowledge_base = load_documents()
print(f"[✓] Found {len(knowledge_base)} knowledge chunks.")

print("[✓] Encoding with SentenceTransformer...")
kb_embeddings = model.encode(knowledge_base, convert_to_numpy=True, show_progress_bar=True)

dimension = kb_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(kb_embeddings)
print("[✓] FAISS index built.")

# Language utils
def translate_to_en(text):
    try:
        lang = detect(text)
        if lang == "en":
            return text.strip(), "en"
        elif lang == "te":
            translated = GoogleTranslator(source='te', target='en').translate(text)
            return translated.strip(), "te"
        else:
            return text.strip(), "en"
    except:
        return text.strip(), "en"

def translate_back(text, lang):
    if lang == "en":
        return text
    try:
        return GoogleTranslator(source='en', target=lang).translate(text)
    except:
        return text

# Search logic
def find_best_paragraph(query_en, top_k=3):
    query_vector = model.encode([query_en], convert_to_numpy=True)
    D, I = index.search(query_vector, top_k)
    matches = []
    for i, score in zip(I[0], D[0]):
        if score < 1.1:
            matches.append((knowledge_base[i], 1 - score))
    return matches

# Final answer logic
def answer_question(query):
    query = query.strip()
    if not query:
        return "Please enter a valid question."

    query_en, lang = translate_to_en(query)

    if not knowledge_base:
        return "No documents loaded."

    matches = find_best_paragraph(query_en)
    if matches:
        best_text, _ = matches[0]
        translated = translate_back(best_text, lang)
        return translated.strip()
    else:
        return "No relevant answer found."

def load_available_languages():
    return ["Auto-detect"]
