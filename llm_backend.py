import os
import faiss
import torch
import numpy as np
from langdetect import detect
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer

# Config
TRANSCRIPTS_DIR = r"D:\llm project\my1"  # Change path if needed
EMBED_DIM = 384
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load model
model = SentenceTransformer(EMBED_MODEL)

# Load and embed all text files
def load_knowledge_base():
    texts, paths = [], []
    for filename in os.listdir(TRANSCRIPTS_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(TRANSCRIPTS_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                texts.append(text)
                paths.append(path)
    embeddings = model.encode(texts, convert_to_tensor=True).cpu().numpy()
    return texts, embeddings

# FAISS index
def build_faiss_index(embeddings):
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(np.array(embeddings))
    return index

# Language helpers
def translate(text, source, target):
    if source == target:
        return text
    try:
        return GoogleTranslator(source=source, target=target).translate(text)
    except:
        return text

# Main: get answer
def get_answer(query):
    input_lang = detect(query)
    query_en = translate(query, input_lang, "en")

    texts, embeddings = load_knowledge_base()
    index = build_faiss_index(embeddings)

    query_vec = model.encode([query_en])[0]
    D, I = index.search(np.array([query_vec]), k=1)
    
    best_text = texts[I[0][0]] if I[0][0] < len(texts) else "Sorry, no relevant answer found."
    best_text_lang = translate(best_text, "en", input_lang)

    return best_text_lang
