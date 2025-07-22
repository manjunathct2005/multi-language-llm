# llm_backend.py

import os
import torch
import numpy as np
import faiss
import re
from langdetect import detect
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util

# === CONFIG ===
TEXT_FOLDER = r"D:\llm project\my1"  # Folder with text transcripts
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# === Load model and initialize ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)

# === Load transcripts and create embeddings ===
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_knowledge_base(folder):
    docs, filenames = [], []
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                content = clean_text(f.read())
                docs.append(content)
                filenames.append(fname)
    return docs, filenames

def get_embeddings(texts):
    return embedder.encode(texts, convert_to_tensor=True)

print("📚 Loading knowledge base...")
docs, filenames = load_knowledge_base(TEXT_FOLDER)
doc_embeddings = get_embeddings(docs)
print("✅ Knowledge base loaded with", len(docs), "documents.")

# === Language Utilities ===
def detect_lang(text):
    return detect(text)

def translate_to_en(text):
    return GoogleTranslator(source='auto', target='en').translate(text)

def translate_back(text, lang):
    return GoogleTranslator(source='en', target=lang).translate(text)

# === Main Answering Function ===
def get_answer(user_query):
    try:
        user_lang = detect_lang(user_query)
        query_en = translate_to_en(user_query)

        query_embedding = embedder.encode(query_en, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]

        top_k = min(1, len(scores))  # return only 1 best result
        top_indices = torch.topk(scores, k=top_k)[1].tolist()
        top_result = docs[top_indices[0]]

        # Return translated answer
        answer_final = translate_back(top_result, user_lang)
        return answer_final

    except Exception as e:
        return f"❌ Error during answer generation: {str(e)}"
