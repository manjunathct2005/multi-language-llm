import os
import re
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
from langdetect import detect
from googletrans import Translator
from sentence_transformers import SentenceTransformer

TEXT_FOLDER = "my1"
MODEL_NAME = "all-MiniLM-L6-v2"

translator = Translator()
model = SentenceTransformer(MODEL_NAME)

def clean_and_chunk(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"---+", "\n---\n", text)
    text = re.sub(r"\n{2,}", "\n", text)

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

def load_documents():
    all_chunks = []
    for file in os.listdir(TEXT_FOLDER):
        if file.endswith(".txt"):
            with open(os.path.join(TEXT_FOLDER, file), encoding="utf-8") as f:
                text = f.read()
                chunks = clean_and_chunk(text)
                all_chunks.extend(chunks)
    return all_chunks

print("[✓] Loading and cleaning text files...")
knowledge_base = load_documents()
print(f"[✓] {len(knowledge_base)} knowledge blocks found.")

print("[✓] Creating embeddings...")
kb_embeddings = model.encode(knowledge_base, convert_to_numpy=True, show_progress_bar=True)

print("[✓] Embeddings ready.")

def translate_to_en(text):
    try:
        text_clean = re.sub(r"[^\w\s]", "", text.lower().strip())
        word_count = len(text_clean.split())

        if word_count <= 3:
            return text.strip(), "en"
        lang = detect(text)
        if lang == "en":
            return text, "en"
        translated = translator.translate(text, src=lang, dest="en")
        return translated.text, lang
    except Exception as e:
        return text, "en"

def get_answer(query):
    query_en, original_lang = translate_to_en(query)
    query_embedding = model.encode([query_en], convert_to_numpy=True)
    similarities = cosine_similarity(query_embedding, kb_embeddings)[0]
    best_match_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_match_idx])
    answer_en = knowledge_base[best_match_idx]

    # Translate back if needed
    if original_lang != "en":
        try:
            answer_translated = translator.translate(answer_en, src="en", dest=original_lang)
            return answer_translated.text
        except Exception:
            return answer_en
    return answer_en

