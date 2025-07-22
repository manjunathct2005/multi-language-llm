# llm_backend.py
import os
import re
import torch
import numpy as np
import faiss
from langdetect import detect
from sentence_transformers import SentenceTransformer
from googletrans import Translator

# === CONFIG ===
TEXT_FOLDER = r"D:\llm project\my1"
EMBEDDING_DIM = 384
MODEL_NAME = "all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === LOAD EMBEDDING MODEL ===
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

# === LOAD FILES ===
def clean_text(text):
    lines = text.splitlines()
    seen = set()
    cleaned = []
    for line in lines:
        line = line.strip()
        if len(line) > 1 and line not in seen:
            seen.add(line)
            cleaned.append(line)
    return " ".join(cleaned)

texts, filenames = [], []
for fname in os.listdir(TEXT_FOLDER):
    if fname.endswith(".txt"):
        with open(os.path.join(TEXT_FOLDER, fname), "r", encoding="utf-8") as f:
            content = f.read().strip()
            cleaned = clean_text(content)
            texts.append(cleaned)
            filenames.append(fname)

# === BUILD EMBEDDING INDEX ===
embeddings = model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)
index = faiss.IndexFlatIP(EMBEDDING_DIM)
index.add(np.array(embeddings))

# === TRANSLATOR ===
translator = Translator()

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def translate(text, src, dest):
    try:
        return translator.translate(text, src=src, dest=dest).text
    except:
        return text

def answer_question(query):
    original_lang = detect_language(query)
    query_en = translate(query, src=original_lang, dest="en")

    q_embed = model.encode([query_en], convert_to_tensor=False, normalize_embeddings=True)
    D, I = index.search(np.array(q_embed), k=1)

    if D[0][0] < 0.3:
        return translate("Sorry, I couldn't find an answer.", src="en", dest=original_lang)

    matched_text = texts[I[0][0]]
    answer_en = highlight_answer(matched_text, query_en)
    final_answer = translate(answer_en, src="en", dest=original_lang)
    return final_answer

def highlight_answer(text, query):
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    return pattern.sub(r'<span style="color:red"><b>\g<0></b></span>', text)
