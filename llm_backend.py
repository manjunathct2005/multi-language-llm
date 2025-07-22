# llm_backend.py

import os
import re
import faiss
import torch
import numpy as np
from langdetect import detect
from googletrans import Translator
from sentence_transformers import SentenceTransformer

# === CONFIG ===
TEXT_FOLDER = r"D:\llm project\my1"  # Change this if needed
EMBEDDING_DIM = 384  # for MiniLM-L6
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# === LOAD EMBEDDING MODEL ===
model = SentenceTransformer(MODEL_NAME)
translator = Translator()

# === LOAD TEXT FILES AND CREATE EMBEDDINGS ===
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", " ", text)
    return text.strip()

def load_and_embed_texts(folder_path):
    sentences = []
    files = os.listdir(folder_path)
    for fname in files:
        if fname.endswith(".txt"):
            with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
                raw = f.read()
                cleaned = clean_text(raw)
                if cleaned:
                    sentences.append(cleaned)

    embeddings = model.encode(sentences, convert_to_numpy=True)
    return sentences, embeddings

sentences, embeddings = load_and_embed_texts(TEXT_FOLDER)

# === CREATE FAISS INDEX ===
index = faiss.IndexFlatL2(EMBEDDING_DIM)
index.add(np.array(embeddings))

# === SEARCH FUNCTION ===
def search_answer(query):
    input_lang = detect(query)
    query_en = translator.translate(query, src=input_lang, dest="en").text

    query_emb = model.encode([query_en])
    D, I = index.search(query_emb, k=1)

    best_match = sentences[I[0][0]]
    translated_ans = translator.translate(best_match, src="en", dest=input_lang).text
    return translated_ans
