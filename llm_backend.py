import os
import re
import torch
import numpy as np
import faiss
from langdetect import detect
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer

# === CONFIG ===
TEXT_FOLDER = r"D:\llm project\my1"  # Replace with your actual path
MODEL_NAME = "all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === LOAD MODEL ===
model = SentenceTransformer(MODEL_NAME, device=DEVICE)

# === TRANSLATION ===
def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        return text

def translate_back(text, target_lang):
    try:
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except Exception:
        return text

# === LOAD & EMBED KNOWLEDGE BASE ===
def clean_text(text):
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"(.)\1{3,}", r"\1", text)
    return text

def load_knowledge_base(folder):
    texts, files = [], []
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            with open(os.path.join(folder, fname), 'r', encoding='utf-8') as f:
                content = f.read()
                texts.append(clean_text(content))
                files.append(fname)
    embeddings = model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))
    return texts, index, files, np.array(embeddings)

kb_texts, kb_index, kb_files, kb_embeddings = load_knowledge_base(TEXT_FOLDER)

# === MAIN FUNCTION ===
def get_answer(question):
    lang = detect(question)
    q_en = translate_to_english(question)
    q_embedding = model.encode([q_en], normalize_embeddings=True)
    D, I = kb_index.search(np.array(q_embedding), k=1)
    top_match = kb_texts[I[0][0]]
    ans = top_match if D[0][0] < 0.4 else "Sorry, I couldn't find a relevant answer."
    return translate_back(ans, lang)
