import os
import torch
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from langdetect import detect
from deep_translator import GoogleTranslator

# Paths
TRANSCRIPTS_DIR = "my1"
EMBEDDINGS_PATH = "D:/hindupur_dataset/embeddings1.pt"

# Load model
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
        embeddings = torch.load(EMBEDDINGS_PATH)
    else:
        embeddings = embedding_model.encode(texts, show_progress_bar=True)
        torch.save(embeddings, EMBEDDINGS_PATH)

    index = NearestNeighbors(n_neighbors=1, metric="cosine")
    index.fit(embeddings)
    return texts, index

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def translate_to_english(text, src_lang):
    if src_lang == "en":
        return text
    return GoogleTranslator(source=src_lang, target="en").translate(text)

def translate_back(text, dest_lang):
    if dest_lang == "en":
        return text
    return GoogleTranslator(source="en", target=dest_lang).translate(text)

def answer_question(query, texts, index):
    query_embedding = embedding_model.encode([query])
    _, indices = index.kneighbors(query_embedding)
    return texts[indices[0][0]]

def process_input(user_input, texts, index):
    source_lang = detect_language(user_input)
    english_query = translate_to_english(user_input, source_lang)
    answer_in_english = answer_question(english_query, texts, index)
    answer_in_original_lang = translate_back(answer_in_english, source_lang)
    return answer_in_original_lang
