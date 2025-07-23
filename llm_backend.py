import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from langdetect import detect
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator

# Paths
TRANSCRIPT_FOLDER = "my1"

# Load SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load knowledge base
knowledge_base = []
knowledge_embeddings = []

for filename in os.listdir(TRANSCRIPT_FOLDER):
    if filename.endswith(".txt"):
        filepath = os.path.join(TRANSCRIPT_FOLDER, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()
            if content.strip():
                knowledge_base.append(content)
                embedding = model.encode(content)
                knowledge_embeddings.append(embedding)

if knowledge_embeddings:
    knowledge_embeddings = np.vstack(knowledge_embeddings)
else:
    knowledge_embeddings = np.empty((0, 384))  # MiniLM-L6-v2 output size

# Translate input to English using deep_translator
def translate_to_english(text, source_lang=None):
    try:
        if not source_lang:
            source_lang = detect(text)
        return GoogleTranslator(source=source_lang, target='en').translate(text)
    except Exception as e:
        print(f"[Translation Error] {e}")
        return text  # fallback

# Translate output back to original language
def translate_to_original(text, target_lang):
    try:
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except Exception as e:
        print(f"[Back Translation Error] {e}")
        return text

# Process user query
def process_input(query):
    original_lang = detect(query)
    translated_query = translate_to_english(query, source_lang=original_lang)

    query_embedding = model.encode(translated_query)
    similarities = cosine_similarity([query_embedding], knowledge_embeddings)

    if similarities.size == 0 or np.max(similarities) < 0.3:
        return "⚠️ No relevant answer found in your knowledge base.", 0.0

    best_idx = int(np.argmax(similarities))
    best_text = knowledge_base[best_idx]
    translated_back = translate_to_original(best_text, target_lang=original_lang)
    confidence = float(np.max(similarities))

    return translated_back, confidence
