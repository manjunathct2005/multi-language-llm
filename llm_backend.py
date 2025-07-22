# llm_backend.py
import os
import torch
import faiss
import numpy as np
from langdetect import detect
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util

# === CONFIG ===
TRANSCRIPT_FOLDER = r"D:\llm project\my1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and cached embeddings
model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
index = faiss.IndexFlatL2(384)
corpus = []
file_names = []

for fname in os.listdir(TRANSCRIPT_FOLDER):
    if fname.endswith(".txt"):
        with open(os.path.join(TRANSCRIPT_FOLDER, fname), 'r', encoding='utf-8') as f:
            text = f.read()
            corpus.append(text)
            file_names.append(fname)

corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
index.add(corpus_embeddings.cpu().detach().numpy())

def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "en"

def translate_to_english(text, src_lang):
    if src_lang == "en":
        return text
    return GoogleTranslator(source=src_lang, target="en").translate(text)

def translate_from_english(text, target_lang):
    if target_lang == "en":
        return text
    return GoogleTranslator(source="en", target=target_lang).translate(text)

def get_answer(question: str) -> str:
    input_lang = detect_language(question)
    translated_question = translate_to_english(question, input_lang)
    question_embedding = model.encode(translated_question, convert_to_tensor=True)
    
    D, I = index.search(question_embedding.cpu().detach().numpy().reshape(1, -1), k=1)
    
    if len(I[0]) == 0:
        return translate_from_english("Sorry, I could not find an answer.", input_lang)

    matched_idx = I[0][0]
    answer = corpus[matched_idx]

    return translate_from_english(answer, input_lang)
