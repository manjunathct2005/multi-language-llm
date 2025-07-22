import os
import glob
import torch
import numpy as np
import faiss
from langdetect import detect
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

# === Configuration ===
TRANSCRIPT_DIR = "my1"  # Folder with .txt transcript files
EMBEDDINGS_PATH = r"D:\hindupur_dataset\embeddings1.pt"
MODEL_NAME = "all-MiniLM-L6-v2"

# === Load the embedding model ===
embedding_model = SentenceTransformer(MODEL_NAME)

# === Translation Functions ===
def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text  # fallback

def translate_from_english(text, target_lang):
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except:
        return text  # fallback

# === Language Detection ===
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# === Clean text from each file ===
def clean_text(text):
    lines = text.splitlines()
    cleaned = [line.strip() for line in lines if line.strip()]
    return "\n".join(cleaned)

# === Load and Embed Text Files ===
def knowledge_base():
    if os.path.exists(EMBEDDINGS_PATH):
        data = torch.load(EMBEDDINGS_PATH)
        texts = data["texts"]
        embeddings = data["embeddings"]
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return texts, index, embeddings

    # Process .txt files
    files = glob.glob(os.path.join(TRANSCRIPT_DIR, "*.txt"))
    texts = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            raw = f.read()
            cleaned = clean_text(raw)
            if cleaned:
                texts.append(cleaned)

    if not texts:
        raise ValueError("No valid .txt files found in transcript directory.")

    # Compute embeddings
    embeddings = embedding_model.encode(texts, convert_to_tensor=False)
    embeddings = np.array(embeddings).astype("float32")

    # FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save for later use
    torch.save({"texts": texts, "embeddings": embeddings}, EMBEDDINGS_PATH)
    return texts, index, embeddings

# === Main Q&A Processing ===
def process_input(query, mode, texts, index, embeddings, lang):
    try:
        query_en = translate_to_english(query) if lang != "en" else query
        query_embedding = embedding_model.encode([query_en], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype("float32")
        D, I = index.search(query_embedding, k=3)

        # Get top matched responses
        results = [texts[i] for i in I[0] if i < len(texts)]

        if not results:
            return translate_from_english("No relevant answer found.", lang)

        if mode == "summary":
            response = results[0]
        else:
            response = "\n\n".join(results)

        return translate_from_english(response, lang) if lang != "en" else response

    except Exception as e:
        return f"⚠️ Error occurred: {str(e)}"
