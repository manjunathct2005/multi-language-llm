# llm_backend.py

import os
import torch
import re
import faiss
import numpy as np
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator

# === CONFIG ===
TEXT_FOLDER = r"D:\llm project\my1"  # Change if needed
EMBEDDING_DIM = 384
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Load embedding model ===
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

# === Helper: Clean Text ===
def clean_text(text):
    return re.sub(r"\s+", " ", text.strip())

# === Load and Embed Transcripts ===
def load_transcripts_and_embed():
    texts, file_names = [], []
    for file in os.listdir(TEXT_FOLDER):
        if file.endswith(".txt"):
            with open(os.path.join(TEXT_FOLDER, file), "r", encoding="utf-8") as f:
                texts.append(clean_text(f.read()))
                file_names.append(file)

    embeddings = embedding_model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    return texts, embeddings, file_names

# === Vector Store ===
docs, doc_embeddings, file_names = load_transcripts_and_embed()

# === Translation ===
def translate_to_english(text, src_lang):
    return GoogleTranslator(source=src_lang, target="en").translate(text)

def translate_from_english(text, target_lang):
    return GoogleTranslator(source="en", target=target_lang).translate(text)

# === Language-Aware Answering ===
def get_answer(query: str) -> str:
    try:
        lang = detect(query)
        translated_query = translate_to_english(query, lang)

        query_embedding = embedding_model.encode(translated_query, convert_to_tensor=True)
        top_k = min(5, len(docs))
        hits = util.semantic_search(query_embedding, doc_embeddings, top_k=top_k)[0]

        for hit in hits:
            answer = docs[hit['corpus_id']]
            translated_answer = translate_from_english(answer, lang)
            return translated_answer

        return translate_from_english("Sorry, I couldn’t find an answer.", lang)

    except Exception as e:
        return f"Error: {str(e)}"
