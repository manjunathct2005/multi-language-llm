import os
import torch
import numpy as np
from langdetect import detect
from googletrans import Translator
from sentence_transformers import SentenceTransformer, util

# CONFIG
TEXT_FOLDER = r"my1"  # just keep your text files here
EMBEDDINGS_CACHE = "embeddings_cache.pt"
MODEL_NAME = "all-MiniLM-L6-v2"

# Load embedding model
model = SentenceTransformer(MODEL_NAME)

# Translator
translator = Translator()

# Load and preprocess text files
def load_texts():
    texts = []
    filenames = []
    for fname in os.listdir(TEXT_FOLDER):
        if fname.endswith(".txt"):
            with open(os.path.join(TEXT_FOLDER, fname), "r", encoding="utf-8") as f:
                content = f.read().strip().replace("\n", " ")
                if content:
                    texts.append(content)
                    filenames.append(fname)
    return texts, filenames

# Generate or load embeddings
def get_embeddings(texts):
    if os.path.exists(EMBEDDINGS_CACHE):
        data = torch.load(EMBEDDINGS_CACHE)
        return data["embeddings"], data["texts"]
    embeddings = model.encode(texts, convert_to_tensor=True)
    torch.save({"embeddings": embeddings, "texts": texts}, EMBEDDINGS_CACHE)
    return embeddings, texts

# Language detection + answer search
def search_answer(query):
    input_lang = detect(query)
    translated_query = translator.translate(query, src=input_lang, dest="en").text

    texts, _ = load_texts()
    if not texts:
        return "No text data found in 'my1' folder."

    embeddings, text_data = get_embeddings(texts)
    query_embedding = model.encode(translated_query, convert_to_tensor=True)

    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    top_idx = int(torch.argmax(cos_scores))

    best_match = text_data[top_idx]
    translated_response = translator.translate(best_match, src="en", dest=input_lang).text
    return translated_response
