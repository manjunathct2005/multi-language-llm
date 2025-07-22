import os
import torch
from sklearn.neighbors import NearestNeighbors
from langdetect import detect
from deep_translator import GoogleTranslator

# Constants
TRANSCRIPTS_DIR = "my1"  # Folder with .txt files
EMBEDDINGS_PATH = "D:/hindupur_dataset/embeddings1.pt"  # Must be precomputed and uploaded
K_NEIGHBORS = 3

# Clean transcript text lines
def clean_text(text):
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if line and not any(line.lower().startswith(x) for x in ("part", "#", "*", "what is data", "datasci")):
            cleaned.append(line)
    return " ".join(cleaned)

# Load .txt files and embeddings
def load_texts_and_embeddings():
    texts = []
    files = sorted([f for f in os.listdir(TRANSCRIPTS_DIR) if f.endswith(".txt")])
    for file in files:
        with open(os.path.join(TRANSCRIPTS_DIR, file), "r", encoding="utf-8") as f:
            content = f.read()
            texts.append(clean_text(content))

    if not os.path.exists(EMBEDDINGS_PATH):
        raise FileNotFoundError(f"Embeddings not found at {EMBEDDINGS_PATH}")
    embeddings = torch.load(EMBEDDINGS_PATH)

    index = NearestNeighbors(n_neighbors=min(K_NEIGHBORS, len(embeddings)), metric="cosine")
    index.fit(embeddings)

    return texts, index

# Translate to English
def translate_to_english(text):
    try:
        return GoogleTranslator(source="auto", target="en").translate(text)
    except:
        return text  # Fallback if translation fails

# Translate back to original language
def translate_from_english(text, target_lang):
    try:
        if len(text.strip()) < 3:
            return "🤖 I couldn't find a meaningful answer."
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except:
        return text  # Fallback if translation fails

# Get nearest answer
def get_answer(query, texts, index, embeddings):
    if not query.strip():
        return "🤖 Please enter a valid question."

    try:
        # Detect language
        query_lang = detect(query)
        english_query = translate_to_english(query)

        # Embed query using existing SentenceTransformer model (already used to create embeddings1.pt)
        from sentence_transformers import SentenceTransformer
        temp_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        query_embedding = temp_model.encode([english_query])

        # Search top-k answers
        distances, indices = index.kneighbors(query_embedding)
        top_indices = indices[0]

        # Combine top answers
        combined_answer = " ".join([texts[i] for i in top_indices])

        # Translate back to original input language
        return translate_from_english(combined_answer, query_lang)
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Public interface
def process_input(question, texts, index):
    embeddings = torch.load(EMBEDDINGS_PATH)
    return get_answer(question, texts, index, embeddings)
