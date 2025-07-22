import os
import glob
import faiss
import torch
import numpy as np
import re
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
from deep_translator import DeepL

TRANSCRIPT_DIR = "my1"
MODEL_NAME = "all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(MODEL_NAME)

def clean_text(text):
    return re.sub(r"(part\s*\d+|what is data science|#|[*•])", "", text, flags=re.I).strip()

def load_texts_and_embeddings():
    texts = []
    for file in glob.glob(os.path.join(TRANSCRIPT_DIR, "*.txt")):
        with open(file, "r", encoding="utf-8") as f:
            lines = [clean_text(line.strip()) for line in f if line.strip()]
            texts.extend(lines)

    embeddings = embedding_model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return texts, index

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def translate_to_english(text):
    try:
        return DeepL(source='auto', target='en').translate(text)
    except:
        return text

def translate_from_english(text, target):
    if target == "en":
        return text
    try:
        return DeepL(source='en', target=target).translate(text)
    except:
        return text

def get_answer(question, texts, index):
    query = clean_text(question)
    query_embedding = embedding_model.encode([query], convert_to_tensor=False, normalize_embeddings=True)
    D, I = index.search(np.array(query_embedding).astype("float32"), k=3)

    best_match = texts[I[0][0]] if I[0][0] < len(texts) else "No relevant answer found."
    return best_match
