# llm_backend.py

import os
import glob
import torch
import numpy as np
import re
from deep_translator import DeeplTranslator
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
import faiss

# === Configuration ===
TRANSCRIPT_DIR = "my1"  # Folder with your .txt files
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDINGS_PATH = os.path.join("D:/hindupur_dataset", "embeddings1.pt")

# === Load the embedding model ===
embedding_model = SentenceTransformer(MODEL_NAME)

# === Translation ===
def translate_to_english(text):
    try:
        return DeeplTranslator(source='auto', target='en').translate(text)
    except:
        return text

def translate_from_english(text, lang):
    if lang == "en":
        return text
    try:
        return DeeplTranslator(source='en', target=lang).translate(text)
    except:
        return text

# === Clean each line ===
def clean_text(text):
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()

        # Skip empty or short lines
        if len(line.split()) < 4:
            continue

        # Remove headers like "Part 1", "Module 3", "Section 4", etc.
        if re.match(r'^(part|chapter|module|section|page)[\s\d:.-]*$', line.lower()):
            continue

        # Remove headings like "What is Data Science", "Overview"
        if re.match(r'^(what is|overview|definition|introduction)', line.lower()):
            continue

        # Remove markdown symbols and unwanted characters
        line = re.sub(r'[#*:\-•►●]', '', line)
        line = re.sub(r'\s+', ' ', line)

        cleaned.append(line)
    return "\n".join(cleaned)

# === Load and clean all text files ===
def load_texts_and_embeddings():
    if os.path.exists(EMBEDDINGS_PATH):
        print("✅ Loading existing embeddings")
        data = torch.load(EMBEDDINGS_PATH)
        return data['texts'], data['index'], data['embeddings']

    print("🔍 Generating new embeddings...")
    texts = []
    for filepath in glob.glob(os.path.join(TRANSCRIPT_DIR, "*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            clean = clean_text(text)
            if clean.strip():
                texts.append(clean)

    embeddings = embedding_model.encode(texts, convert_to_tensor=False)
    embeddings_np = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)

    torch.save({'texts': texts, 'index': index, 'embeddings': embeddings_np}, EMBEDDINGS_PATH)
    return texts, index, embeddings_np

# === Detect language ===
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# === Process question ===
def process_input(user_input, texts, index):
    lang = detect_language(user_input)
    question_en = translate_to_english(user_input)
    query_vec = embedding_model.encode([question_en])[0].astype("float32")

    D, I = index.search(np.array([query_vec]), k=1)
    if D[0][0] > 1.2:  # distance threshold
        return translate_from_english("Sorry, I couldn't find a relevant answer.", lang)

    best_answer = texts[I[0][0]]
    return translate_from_english(best_answer, lang)
