import os
import torch
import numpy as np
from langdetect import detect
from googletrans import Translator
from sentence_transformers import SentenceTransformer, util

# === CONFIG ===
TEXT_FOLDER = "D:/llm project/my1"  # Change if needed
EMBEDDING_FILE = "D:/llm project/embeddings1.pt"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# === Load model ===
model = SentenceTransformer(MODEL_NAME)

# === Load or compute embeddings ===
def load_knowledge_base():
    texts, files = [], []
    for fname in os.listdir(TEXT_FOLDER):
        if fname.endswith(".txt"):
            path = os.path.join(TEXT_FOLDER, fname)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                texts.append(content)
                files.append(fname)

    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    return texts, embeddings, files

if os.path.exists(EMBEDDING_FILE):
    kb_texts, kb_embeddings, kb_files = torch.load(EMBEDDING_FILE)
else:
    kb_texts, kb_embeddings, kb_files = load_knowledge_base()
    torch.save((kb_texts, kb_embeddings, kb_files), EMBEDDING_FILE)

# === Translator ===
translator = Translator()

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def translate_to_english(text):
    lang = detect_language(text)
    if lang != "en":
        return translator.translate(text, src=lang, dest="en").text, lang
    return text, "en"

def translate_from_english(text, target_lang):
    if target_lang != "en":
        return translator.translate(text, src="en", dest=target_lang).text
    return text

# === Answer search ===
def search_answer(query):
    query_en, original_lang = translate_to_english(query)
    query_embedding = model.encode(query_en, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, kb_embeddings)[0]
    top_idx = torch.argmax(scores).item()
    best_score = scores[top_idx].item()

    if best_score < 0.4:
        return translate_from_english("Sorry, I couldn't find a relevant answer.", original_lang)

    response = kb_texts[top_idx]
    return translate_from_english(response, original_lang)
