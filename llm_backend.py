import os
import torch
from sentence_transformers import SentenceTransformer, util
from langdetect import detect
from deep_translator import GoogleTranslator
import re

# Load model once globally
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def clean_text(text):
    # Remove emojis, symbols, URLs, multiple spaces
    text = re.sub(r"http\S+", "", text)                       # URLs
    text = re.sub(r"[^\w\s.,!?]", "", text)                   # Special characters
    text = re.sub(r"\n+", " ", text)                          # Newlines
    text = re.sub(r"\s+", " ", text).strip()                  # Extra spaces

    # Remove repeated segments (like ads, names)
    lines = list(set(text.split('.')))
    cleaned = '. '.join([line.strip() for line in lines if len(line.strip()) > 15])
    return cleaned

def load_transcripts(transcript_dir="my1"):
    texts, embeddings = [], []
    seen_texts = set()

    for filename in os.listdir(transcript_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(transcript_dir, filename), "r", encoding="utf-8") as f:
                raw = f.read()
                cleaned = clean_text(raw)

                if cleaned and cleaned not in seen_texts:
                    seen_texts.add(cleaned)
                    texts.append(cleaned)
                    embeddings.append(embedder.encode(cleaned, convert_to_tensor=True))
    
    return texts, embeddings

def translate_to_en(text, src_lang):
    if src_lang == "en":
        return text
    return GoogleTranslator(source=src_lang, target="en").translate(text)

def translate_from_en(text, target_lang):
    if target_lang == "en":
        return text
    return GoogleTranslator(source="en", target=target_lang).translate(text)

def answer_question(question, texts, embeddings):
    input_lang = detect(question)
    translated_q = translate_to_en(question, input_lang)

    q_embedding = embedder.encode(translated_q, convert_to_tensor=True)
    scores = util.cos_sim(q_embedding, torch.stack(embeddings))[0]

    best_idx = torch.argmax(scores).item()
    best_answer = texts[best_idx]

    return translate_from_en(best_answer, input_lang)
