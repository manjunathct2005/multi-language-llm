import os
import torch
from sentence_transformers import SentenceTransformer, util
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer

TRANSCRIPTS_DIR = "my1"
EMBEDDINGS_PATH = "embeddings1.pt"

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_transcripts():
    texts = []
    for file in os.listdir(TRANSCRIPTS_DIR):
        if file.endswith(".txt"):
            with open(os.path.join(TRANSCRIPTS_DIR, file), "r", encoding="utf-8") as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines if len(line.strip()) > 10]
                texts.extend(lines)
    return texts

def translate(text, src, tgt):
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        translation_model = MarianMTModel.from_pretrained(model_name)
        tokens = tokenizer(text, return_tensors="pt", padding=True)
        translated = translation_model.generate(**tokens)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except:
        return text

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def load_texts_and_embeddings():
    texts = load_transcripts()
    if os.path.exists(EMBEDDINGS_PATH):
        embeddings = torch.load(EMBEDDINGS_PATH)
    else:
        embeddings = model.encode(texts, convert_to_tensor=True)
        torch.save(embeddings, EMBEDDINGS_PATH)
    return texts, embeddings

texts, embeddings = load_texts_and_embeddings()

def answer_question(question):
    lang = detect_language(question)
    translated_q = question if lang == "en" else translate(question, lang, "en")
    query_embedding = model.encode(translated_q, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, embeddings)[0]
    best_score = torch.max(scores).item()
    best_index = torch.argmax(scores).item()

    if best_score < 0.5:
        response = "No relevant answer found."
    else:
        response = texts[best_index]

    return response if lang == "en" else translate(response, "en", lang)
