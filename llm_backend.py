import os
import torch
import faiss
import numpy as np
from langdetect import detect
from googletrans import Translator
from sentence_transformers import SentenceTransformer

# === CONFIG ===
TEXT_FOLDER = r"D:\llm project\my1"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
translator = Translator()

# === Load Embedding Model ===
embedding_model = SentenceTransformer(MODEL_NAME)

# === Load Knowledge Base ===
corpus = []
file_names = []

for file in os.listdir(TEXT_FOLDER):
    if file.endswith(".txt"):
        file_path = os.path.join(TEXT_FOLDER, file)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            corpus.extend(lines)
            file_names.extend([file] * len(lines))

# === Generate Embeddings ===
corpus_embeddings = embedding_model.encode(corpus, convert_to_tensor=False)
corpus_embeddings = np.array(corpus_embeddings).astype("float32")

# === Create FAISS Index ===
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings)

# === Language Utilities ===
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"  # fallback to English

def translate(text, src, dest):
    if src == dest:
        return text
    try:
        return translator.translate(text, src=src, dest=dest).text
    except:
        return text

# === Search Function ===
def search_embeddings(query, top_k=3):
    query_embedding = embedding_model.encode([query])[0].astype("float32")
    distances, indices = index.search(np.array([query_embedding]), top_k)
    
    results = []
    for idx in indices[0]:
        if idx < len(corpus):
            results.append(corpus[idx])
    
    # Format the answer as clean paragraphs
    return "\n\n".join(f"• {r}" for r in results if r.strip())

# === Main Entry Point ===
def process_input(user_input):
    input_lang = detect_language(user_input)
    
    # Step 1: Translate to English for semantic search
    user_input_en = translate(user_input, src=input_lang, dest='en')
    
    # Step 2: Retrieve answer from KB
    answer_en = search_embeddings(user_input_en)
    
    # Step 3: Translate back to original language (if needed)
    final_answer = translate(answer_en, src='en', dest=input_lang)

    return final_answer or "❌ Sorry, no relevant answer found in your documents."
