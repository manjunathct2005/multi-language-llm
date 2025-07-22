# llm_backend.py

import os
import glob
import torch
import numpy as np
import re
import faiss
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator

# === Configuration ===
TRANSCRIPT_DIR = "my1"  # Folder with your .txt files
EMBEDDINGS_PATH = r"D:\hindupur_dataset\embeddings1.pt"
MODEL_NAME = "all-MiniLM-L6-v2"

# === Load the embedding model ===
embedding_model = SentenceTransformer(MODEL_NAME)

# === Translation Functions ===
def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        return text

def translate_from_english(text, target_lang):
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except Exception:
        return text

# === Clean Text Function ===
def clean_text(raw):
    raw = raw.lower()
    raw = re.sub(r"(part|section|chapter)\s*\d+.*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"(overview|detailed|summary|key points).*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"[#*•→►]+", " ", raw)  # Remove markdown bullets or headers
    raw = re.sub(r"\d+\s*[.)-]", "", raw)  # Remove numbered list items
    raw = re.sub(r"\s{2,}", " ", raw)  # Extra spaces
    return raw.strip()

# === Load and Embed Text Files ===
def load_texts_and_embeddings():
    if os.path.exists(EMBEDDINGS_PATH):
        data = torch.load(EMBEDDINGS_PATH)
        return data['texts'], data['index']

    texts = []
    for file in glob.glob(os.path.join(TRANSCRIPT_DIR, "*.txt")):
        with open(file, 'r', encoding='utf-8') as f:
            content = f.read()
            cleaned = clean_text(content)
            if cleaned:
                texts.append(cleaned)

    embeddings = embedding_model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    embeddings_np = embeddings.cpu().detach().numpy()
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)

    torch.save({'texts': texts, 'index': index}, EMBEDDINGS_PATH)
    return texts, index

# === Main Q&A Function ===
def process_input(question, texts, index):
    source_lang = detect(question)
    translated_q = translate_to_english(question)
    question_embedding = embedding_model.encode(translated_q, convert_to_tensor=True)
    question_vector = question_embedding.cpu().detach().numpy().reshape(1, -1)

    _, idx = index.search(question_vector, 1)
    answer = texts[idx[0][0]]
    final_answer = translate_from_english(answer, source_lang)
    return final_answer
