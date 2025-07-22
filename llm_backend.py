import os
import torch
import numpy as np
from langdetect import detect
from googletrans import Translator
from sentence_transformers import SentenceTransformer, util

# === CONFIG ===
TEXT_FOLDER = "my1"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# === LOAD MODEL ===
model = SentenceTransformer(EMBEDDING_MODEL)

# === LOAD & EMBED DOCUMENTS ===
documents = []
file_names = []

for fname in os.listdir(TEXT_FOLDER):
    if fname.endswith(".txt"):
        path = os.path.join(TEXT_FOLDER, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            documents.append(text)
            file_names.append(fname)

document_embeddings = model.encode(documents, convert_to_tensor=True)

# === TRANSLATOR ===
translator = Translator()

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def translate_to_english(text, lang):
    if lang == "en":
        return text
    return translator.translate(text, src=lang, dest="en").text

def translate_back(text, lang):
    if lang == "en":
        return text
    return translator.translate(text, src="en", dest=lang).text

def search_answer(query):
    input_lang = detect_language(query)
    query_en = translate_to_english(query, input_lang)
    
    query_embedding = model.encode(query_en, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, document_embeddings)[0]

    best_idx = torch.argmax(scores).item()
    best_score = scores[best_idx].item()

    if best_score < 0.3:
        return translate_back("Sorry, I couldn't find a good answer.", input_lang)

    best_answer = documents[best_idx]
    return translate_back(best_answer.strip(), input_lang)
