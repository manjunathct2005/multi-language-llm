import os
import torch
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from langdetect import detect
from deep_translator import GoogleTranslator

# Paths
TRANSCRIPTS_DIR = "my1"
EMBEDDINGS_PATH = "embeddings1.pt"

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def clean_text(text):
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if line and not any(line.lower().startswith(x) for x in ("part", "#", "*", "what is data", "datasci")):
            cleaned.append(line)
    return " ".join(cleaned)

def load_texts_and_embeddings():
    texts = []
    files = sorted(os.listdir(TRANSCRIPTS_DIR))
    for file in files:
        if file.endswith(".txt"):
            with open(os.path.join(TRANSCRIPTS_DIR, file), "r", encoding="utf-8") as f:
                content = f.read()
                texts.append(clean_text(content))

    if os.path.exists(EMBEDDINGS_PATH):
        try:
            embeddings = torch.load(EMBEDDINGS_PATH)
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            embeddings = embedding_model.encode(texts, show_progress_bar=True)
            torch.save(embeddings, EMBEDDINGS_PATH)
    else:
        embeddings = embedding_model.encode(texts, show_progress_bar=True)
        torch.save(embeddings, EMBEDDINGS_PATH)

    return texts, embeddings

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def translate_to_english(text, src_lang):
    if src_lang != "en":
        return GoogleTranslator(source=src_lang, target="en").translate(text)
    return text

def translate_from_english(text, target_lang):
    if target_lang != "en":
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    return text

def get_answer(question, texts, embeddings, top_k=3):
    question_lang = detect_language(question)
    question_en = translate_to_english(question, question_lang)

    question_embedding = embedding_model.encode([question_en])
    knn = NearestNeighbors(n_neighbors=top_k, metric="cosine")
    knn.fit(embeddings)
    distances, indices = knn.kneighbors(question_embedding)

    results = [texts[idx] for idx in indices[0]]
    combined_answer = " ".join(results)
    return translate_from_english(combined_answer, question_lang)

# ✅ Add this function so app.py works
def process_input(question, texts, embeddings):
    return get_answer(question, texts, embeddings)
