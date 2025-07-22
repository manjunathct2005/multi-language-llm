import os
import torch
import re
from deep_translator import GoogleTranslator  # Offline-friendly
from langdetect import detect
from sentence_transformers import SentenceTransformer, util

# === CONFIG ===
TEXT_FOLDER = r"D:\llm project\my1"  # Make sure this path exists and has .txt transcript files
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME).to(DEVICE)

# === GLOBAL CACHES ===
CHUNKS = []
SOURCES = []
EMBEDDINGS = []

def clean_text(text):
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_transcripts():
    global CHUNKS, SOURCES, EMBEDDINGS
    for file in os.listdir(TEXT_FOLDER):
        if file.endswith(".txt"):
            with open(os.path.join(TEXT_FOLDER, file), 'r', encoding='utf-8') as f:
                content = f.read()
                cleaned = clean_text(content)
                if cleaned:
                    CHUNKS.append(cleaned)
                    SOURCES.append(file)
    EMBEDDINGS.extend(EMBEDDING_MODEL.encode(CHUNKS, convert_to_tensor=True))

def detect_lang(text):
    try:
        return detect(text)
    except:
        return 'en'

def translate(text, src_lang, tgt_lang):
    try:
        return GoogleTranslator(source=src_lang, target=tgt_lang).translate(text)
    except:
        return text  # fallback

def answer_question(query):
    if not EMBEDDINGS:
        load_transcripts()
    query_lang = detect_lang(query)
    query_en = translate(query, src_lang=query_lang, tgt_lang="en")

    query_embedding = EMBEDDING_MODEL.encode(query_en, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, EMBEDDINGS)[0]
    top_idx = torch.argmax(scores).item()
    best_match = CHUNKS[top_idx]
    answer_translated = translate(best_match, src_lang="en", tgt_lang=query_lang)
    return answer_translated

def load_available_languages():
    return ['en', 'hi', 'te', 'kn', 'ta', 'ml']
