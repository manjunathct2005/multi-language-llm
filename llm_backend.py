import os
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from langdetect import detect
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator

# Paths
TRANSCRIPT_FOLDER = "my1"

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")
translator = Translator()

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
    knowledge_embeddings = np.empty((0, 384))  # Default size for MiniLM-L6

# Translate input to English
def translate_to_english(text):
    try:
        return translator.translate(text, dest='en').text
    except Exception:
        return text  # Fallback if translation fails

# Translate output back to original language
def translate_to_original(text, lang_code):
    try:
        return translator.translate(text, dest=lang_code).text
    except Exception:
        return text

# Process user query
def process_input(query):
    original_lang = detect(query)
    translated_query = translate_to_english(query)

    query_embedding = model.encode(translated_query)
    similarities = cosine_similarity([query_embedding], knowledge_embeddings)

    if similarities.size == 0 or np.max(similarities) < 0.3:
        return "No relevant answer found in your knowledge base.", 0.0

    best_idx = int(np.argmax(similarities))
    best_text = knowledge_base[best_idx]
    translated_back = translate_to_original(best_text, original_lang)

    return translated_back, round(float(np.max(similarities)), 2)
