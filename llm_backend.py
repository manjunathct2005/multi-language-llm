# llm_backend.py

import os
import glob
import torch
import numpy as np
import re
import faiss
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator

# === Configuration ===
TRANSCRIPT_DIR = r"D:\hindupur_dataset\my1"  # Folder with your .txt files
EMBEDDINGS_PATH = r"D:\hindupur_dataset\embeddings1.pt"
MODEL_NAME = "all-MiniLM-L6-v2"

# === Load the embedding model (for text similarity) ===
embedding_model = SentenceTransformer(MODEL_NAME)

# === Translation Functions ===
def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        return text

def translate_from_english(text, target_lang):
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except Exception as e:
        return text

# === Language Detection ===
def detect_language(text):
    try:
        lang = detect(text)
        # Support only English, Telugu, Hindi; fallback to English if not one of these
        if lang in ["en", "te", "hi"]:
            return lang
        else:
            return "en"
    except Exception:
        return "en"

# === Clean Text ===
def clean_text(text):
    # Remove extra spaces and newlines
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# === Load Knowledge Base ===
def load_knowledge_base(transcript_dir=TRANSCRIPT_DIR):
    texts = []
    for file_path in glob.glob(os.path.join(transcript_dir, "*.txt")):
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
                cleaned = clean_text(raw)
                if cleaned:
                    texts.append(cleaned)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return texts

# Load the text files from your folder
knowledge_texts = load_knowledge_base()

# === Build Embeddings and FAISS Index ===
if knowledge_texts:
    print("[✓] Generating embeddings for knowledge base...")
    embeddings = embedding_model.encode(knowledge_texts, convert_to_tensor=False, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
else:
    embeddings = None
    faiss_index = None

# Optionally save the embeddings to disk
def save_embeddings(file_path=EMBEDDINGS_PATH):
    if embeddings is not None and knowledge_texts:
        data = {"texts": knowledge_texts, "embeddings": embeddings, "index": faiss_index}
        torch.save(data, file_path)

save_embeddings()

# === Main Processing Function ===
def process_input(query):
    query = query.strip()
    if not query:
        return "Please enter a valid question.", "en"

    # Detect query language (only en, te, hi supported)
    original_lang = detect_language(query)
    # Translate query to English if necessary
    query_en = query if original_lang == "en" else translate_to_english(query)

    if faiss_index is None or embeddings is None or not knowledge_texts:
        return "Knowledge base is empty.", original_lang

    # Encode the query and search in the FAISS index
    query_embedding = embedding_model.encode([query_en], convert_to_tensor=False)
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = faiss_index.search(query_embedding, k=1)
    best_idx = indices[0][0]
    best_distance = float(distances[0][0])
    # Lower L2 distance means higher semantic similarity.
    # Convert distance to an approximate "confidence" (e.g., 1 / (1+distance))
    confidence = 1 / (1 + best_distance)

    # Define a threshold—if distance is too high, no good match was found.
    THRESHOLD = 1.0  # Adjust as needed based on your data
    if best_distance > THRESHOLD:
        return "No relevant answer found in the knowledge base.", original_lang

    best_text = knowledge_texts[best_idx]
    # If the original query language isn't English, translate the answer back
    if original_lang != "en":
        best_text = translate_from_english(best_text, original_lang)
    
    return best_text, f"{confidence:.2f}"
