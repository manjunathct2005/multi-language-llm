import os
import re
import torch
import faiss
import numpy as np
from langdetect import detect
from googletrans import Translator
from sentence_transformers import SentenceTransformer

# === CONFIG ===
TEXT_FOLDER = "my1"  # Folder with text files
MODEL_NAME = "all-MiniLM-L6-v2"
translator = Translator()
model = SentenceTransformer(MODEL_NAME)

# === Clean and Chunk Logic ===
def clean_and_chunk(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"---+", "\n---\n", text)
    text = re.sub(r"\n{2,}", "\n", text)

    chunks = re.split(r"\n\s*---\s*\n", text)
    return [c.strip() for c in chunks if len(c.strip()) > 50]

# === Load documents
def load_documents():
    all_chunks = []
    if not os.path.exists(TEXT_FOLDER):
        return []
    for file in os.listdir(TEXT_FOLDER):
        if file.endswith(".txt"):
            with open(os.path.join(TEXT_FOLDER, file), encoding="utf-8") as f:
                text = f.read()
                chunks = clean_and_chunk(text)
                all_chunks.extend(chunks)
    return all_chunks

# === Initialize knowledge base
knowledge_base = load_documents()
if knowledge_base:
    kb_embeddings = model.encode(knowledge_base, convert_to_numpy=True, show_progress_bar=True)
    index = faiss.IndexFlatL2(kb_embeddings.shape[1])
    index.add(kb_embeddings)
else:
    index = None

# === Translation
def translate_to_en(text):
    try:
        lang = detect(text)
        if lang == "en":
            return text, "en"
        elif lang == "te":
            translated = translator.translate(text, src="te", dest="en")
            return translated.text.strip(), "te"
        else:
            return text, "en"
    except:
        return text, "en"

def translate_back(text, lang):
    try:
        if lang == "en":
            return text
        return translator.translate(text, src="en", dest=lang).text
    except:
        return text

# === Search
def find_best_paragraph(query_en, top_k=3):
    query_vector = model.encode([query_en], convert_to_numpy=True)
    D, I = index.search(query_vector, top_k)
    results = [(knowledge_base[i], 1 - D[0][j]) for j, i in enumerate(I[0])]
    return results

# === Main Logic
def process_input(query):
    query = query.strip()
    if not query:
        return "Please enter a question.", "en"
    if not knowledge_base or index is None:
        return "Knowledge base is empty.", "en"

    query_en, lang = translate_to_en(query)
    matches = find_best_paragraph(query_en)

    if matches:
        best_text, score = matches[0]
        return translate_back(best_text, lang), f"{score:.2f}"
    return "No relevant answer found.", lang
