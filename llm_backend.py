import os
import torch
from sentence_transformers import SentenceTransformer, util
from langdetect import detect
from deep_translator import GoogleTranslator
import re

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?]", "", text)
    lines = list(set(text.split('.')))
    return '. '.join([line.strip() for line in lines if len(line.strip()) > 20])

def load_transcripts(transcript_dir="D:/hindupur_dataset/transcripts"):
    texts, embeddings = [], []
    for filename in os.listdir(transcript_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(transcript_dir, filename), "r", encoding="utf-8") as f:
                content = f.read()
                cleaned = clean_text(content)
                if cleaned:
                    texts.append(cleaned)
                    emb = embedder.encode(cleaned, convert_to_tensor=True)
                    embeddings.append(emb)
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
    question_en = translate_to_en(question, input_lang)
    question_embedding = embedder.encode(question_en, convert_to_tensor=True)

    similarities = util.cos_sim(question_embedding, torch.stack(embeddings))[0]
    best_idx = torch.argmax(similarities).item()
    answer_en = texts[best_idx]

    return translate_from_en(answer_en, input_lang)
