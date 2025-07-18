import os
import re
import faiss
import torch
import numpy as np
from langdetect import detect
from googletrans import Translator
from sentence_transformers import SentenceTransformer

# === CONFIG ===
TEXT_FOLDER = r"D:\hindupur_dataset\my1"  # Your text transcript folder
MODEL_NAME = "all-MiniLM-L6-v2"
translator = Translator()
model = SentenceTransformer(MODEL_NAME)

# === Clean and Chunk Logic ===
def clean_and_chunk(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove emojis/non-ASCII
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"---+", "\n---\n", text)  # Normalize topic markers
    text = re.sub(r"\n{2,}", "\n", text)

    # Split using topic boundaries
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

# === Load documents and clean
def load_documents():
    all_chunks = []
    for file in os.listdir(TEXT_FOLDER):
        if file.endswith(".txt"):
            file_path = os.path.join(TEXT_FOLDER, file)
            try:
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()
                    chunks = clean_and_chunk(text)
                    all_chunks.extend(chunks)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return all_chunks

# === Load and index knowledge base
print("[✓] Loading and cleaning text files...")
knowledge_base = load_documents()
print(f"[✓] {len(knowledge_base)} knowledge blocks found.")

print("[✓] Creating embeddings...")
kb_embeddings = model.encode(knowledge_base, convert_to_numpy=True, show_progress_bar=True)

dimension = kb_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(kb_embeddings)
print("[✓] FAISS index ready.")

# === Language Translate Logic
def translate_to_en(text):
    try:
        text_clean = re.sub(r"[^\w\s]", "", text.lower().strip())
        word_count = len(text_clean.split())

        if word_count <= 3:
            return text.strip(), "en"  # Assume English if too short

        lang = detect(text_clean)
        if lang == "en":
            return text.strip(), "en"
        elif lang == "te":
            translated = translator.translate(text, src="te", dest="en")
            return translated.text.strip(), "te"
        else:
            return text.strip(), "en"  # Fallback to English for unsupported
    except Exception as e:
        return text.strip(), "en"  # On failure, fallback to English

def translate_back(text, lang):
    if lang == "en":
        return text
    try:
        translated = translator.translate(text, src="en", dest=lang)
        return translated.text.strip()
    except:
        return text

# === Semantic Search
def find_best_paragraph(query_en, top_k=3):
    query_vector = model.encode([query_en], convert_to_numpy=True)
    D, I = index.search(query_vector, top_k)
    matches = []
    for i, score in zip(I[0], D[0]):
        if score < 1.1:  # tighter threshold for semantic match
            matches.append((knowledge_base[i], 1 - score))
    return matches

# === Main Q&A Handler
def process_input(query):
    query = query.strip()
    if not query:
        return "Please enter a valid question.", "en"

    query_en, lang = translate_to_en(query)
    if not knowledge_base:
        return "Knowledge base is empty. Please upload transcript files.", lang

    matches = find_best_paragraph(query_en, top_k=3)
    if matches:
        best_text, score = matches[0]
        translated = translate_back(best_text, lang)
        return translated.strip(), f"{score:.2f}"
    else:
        return "No relevant answer found in the documents.", lang
