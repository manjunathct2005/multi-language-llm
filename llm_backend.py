import os
import torch
import faiss
import numpy as np
from langdetect import detect
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer

# === CONFIG ===
TEXT_FOLDER = r"D:\llm project\my1"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load model
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# === Utility Functions ===
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def translate_to_english(text, src_lang):
    if src_lang == "en":
        return text
    try:
        return GoogleTranslator(source=src_lang, target="en").translate(text)
    except:
        return text

def translate_to_original(text, target_lang):
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except:
        return text

def load_transcripts():
    texts = []
    sources = []
    for file in os.listdir(TEXT_FOLDER):
        if file.endswith(".txt"):
            path = os.path.join(TEXT_FOLDER, file)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                texts.append(content)
                sources.append(file)
    return texts, sources

def embed_texts(texts):
    return model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)

def build_faiss_index(embeddings):
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(embeddings))
    return index

# === Load Transcripts and Embeddings ===
CHUNKS, SOURCES = load_transcripts()
CHUNK_EMBEDDINGS = embed_texts(CHUNKS)
INDEX = build_faiss_index(CHUNK_EMBEDDINGS)

# === Main Q&A ===
def answer_question(user_question):
    input_lang = detect_language(user_question)
    english_question = translate_to_english(user_question, input_lang)

    question_embedding = model.encode([english_question], convert_to_tensor=False, normalize_embeddings=True)
    D, I = INDEX.search(np.array(question_embedding), k=1)

    best_idx = I[0][0]
    score = D[0][0]

    if score < 0.4:
        return None, None

    answer_english = CHUNKS[best_idx]
    answer_translated = translate_to_original(answer_english, input_lang)

    return answer_translated.strip(), SOURCES[best_idx]
