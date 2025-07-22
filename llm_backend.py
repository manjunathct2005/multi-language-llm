# llm_backend.py
import os
import torch
import numpy as np
from langdetect import detect
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util

TRANSCRIPTS_FOLDER = r"D:\llm project\my1"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load model once
model = SentenceTransformer(MODEL_NAME)

# Load all embeddings
def load_knowledge_base():
    knowledge = []
    for file in os.listdir(TRANSCRIPTS_FOLDER):
        if file.endswith(".txt"):
            path = os.path.join(TRANSCRIPTS_FOLDER, file)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                sentences = content.split('.')
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence:
                        emb = model.encode(sentence)
                        knowledge.append((sentence, emb))
    return knowledge

knowledge_base = load_knowledge_base()

# Language detection
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# Translation
def translate(text, source, target):
    if source == target:
        return text
    try:
        return GoogleTranslator(source=source, target=target).translate(text)
    except:
        return text

# Get Answer
def get_answer(query):
    user_lang = detect_language(query)
    query_en = translate(query, source=user_lang, target="en")
    query_emb = model.encode(query_en)

    scores = []
    for sent, emb in knowledge_base:
        score = util.cos_sim(query_emb, emb)
        scores.append((sent, score.item()))

    best = sorted(scores, key=lambda x: x[1], reverse=True)[:1]
    if best and best[0][1] > 0.45:
        answer_en = best[0][0]
        answer_final = translate(answer_en, source="en", target=user_lang)
    else:
        answer_final = translate("No relevant answer found in knowledge base.", source="en", target=user_lang)

    return answer_final
