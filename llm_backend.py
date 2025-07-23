import os
import re
import torch
import numpy as np
from langdetect import detect
from googletrans import Translator
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configuration
TEXT_FOLDER = "my1"
MODEL_NAME = "all-MiniLM-L6-v2"

# Load translation and embedding model
translator = Translator()
model = SentenceTransformer(MODEL_NAME)

# Text cleaning and chunking
def clean_and_chunk(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"---+", "\n---\n", text)
    text = re.sub(r"\n{2,}", "\n", text)

    raw_chunks = re.split(r"\n\s*---\s*\n", text)
    clean_chunks = []

    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk or len(chunk) < 50:
            continue
        lines = chunk.split("\n")
        cleaned = "\n".join([line.strip() for line in lines if line.strip()])
        clean_chunks.append(cleaned)
    
    return clean_chunks

# Load and embed documents
def load_documents():
    all_chunks = []
    for file in os.listdir(TEXT_FOLDER):
        if file.endswith(".txt"):
            with open(os.path.join(TEXT_FOLDER, file), encoding="utf-8") as f:
                text = f.read()
                chunks = clean_and_chunk(text)
                all_chunks.extend(chunks)
    return all_chunks

print("[✓] Loading and cleaning text files...")
knowledge_base = load_documents()
print(f"[✓] {len(knowledge_base)} knowledge blocks found.")

print("[✓] Creating embeddings...")
kb_embeddings = model.encode(knowledge_base, convert_to_numpy=True, show_progress_bar=True)

# Translate to English if needed
def translate_to_en(text):
    try:
        detected_lang = detect(text)
        if detected_lang != "en":
            translated = translator.translate(text, src=detected_lang, dest="en").text
            return translated.strip(), detected_lang
        return text.strip(), "en"
    except:
        return text.strip(), "en"

# Translate back to original language
def translate_back_to_lang(text, original_lang):
    try:
        if original_lang != "en":
            translated = translator.translate(text, src="en", dest=original_lang).text
            return translated.strip()
        return text.strip()
    except:
        return text.strip()

# Find best matching knowledge blocks using cosine similarity
def find_best_match(query_embedding, kb_embeddings, top_k=1):
    similarities = cosine_similarity([query_embedding], kb_embeddings)[0]
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    return top_k_indices, similarities[top_k_indices]

# Core processor
def process_input(query_text):
    if not query_text or len(query_text.strip()) < 3:
        return "Empty or too short input provided.", None

    query_en, detected_lang = translate_to_en(query_text)
    query_embedding = model.encode([query_en], convert_to_numpy=True)[0]

    top_k_indices, similarities = find_best_match(query_embedding, kb_embeddings, top_k=1)

    if similarities[0] < 0.45:
        return "⚠️ No relevant answer found in the knowledge base.", similarities[0]

    best_chunk = knowledge_base[top_k_indices[0]]
    answer = translate_back_to_lang(best_chunk, detected_lang)

    return answer, f"{similarities[0]:.2f}"
