import os
import torch
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from langdetect import detect
from deep_translator import GoogleTranslator, exceptions as dt_exceptions

# Paths
TRANSCRIPTS_DIR = "my1"
EMBEDDINGS_PATH = "D:/hindupur_dataset/embeddings1.pt"

# Load embedding model
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

    return texts, embeddings

def translate_to_english(text):
    try:
        lang = detect(text)
        if lang != "en":
            return GoogleTranslator(source=lang, target="en").translate(text), lang
        else:
            return text, "en"
    except dt_exceptions.NotValidPayload:
        return text, "en"
    except Exception:
        return text, "en"

def translate_from_english(text, target_lang):
    try:
        if len(text) > 4900:
            text = text[:4900]
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except dt_exceptions.NotValidLength:
        return text
    except Exception:
        return text

def get_answer(question, texts, embeddings):
    question_emb = embedding_model.encode([question])
    knn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(embeddings)
    _, indices = knn.kneighbors(question_emb)

    matched_texts = [texts[i] for i in indices[0]]
    combined_answer = "\n".join(matched_texts)

    question_lang = detect(question)
    if question_lang != "en":
        return translate_from_english(combined_answer, question_lang)
    else:
        return combined_answer

def process_input(question, texts, embeddings):
    question_en, lang = translate_to_english(question)
    return get_answer(question_en, texts, embeddings)
