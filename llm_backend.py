import os
import torch
import numpy as np
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer
from sentence_transformers import SentenceTransformer, util

# === CONFIG ===
TEXT_FOLDER = r"D:\llm project\my1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(model_name, device=DEVICE)

# === TRANSLATION MODELS ===
LANGUAGE_MODELS = {
    "hi": "Helsinki-NLP/opus-mt-hi-en",
    "te": "Helsinki-NLP/opus-mt-te-en",
    "kn": "Helsinki-NLP/opus-mt-kn-en",
    "en": None  # No translation needed
}

REVERSE_MODELS = {
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "te": "Helsinki-NLP/opus-mt-en-te",
    "kn": "Helsinki-NLP/opus-mt-en-kn",
    "en": None
}

def load_model_pair(lang):
    src = LANGUAGE_MODELS.get(lang)
    tgt = REVERSE_MODELS.get(lang)
    src_model = src_tokenizer = tgt_model = tgt_tokenizer = None

    if src:
        src_model = MarianMTModel.from_pretrained(src).to(DEVICE)
        src_tokenizer = MarianTokenizer.from_pretrained(src)
    if tgt:
        tgt_model = MarianMTModel.from_pretrained(tgt).to(DEVICE)
        tgt_tokenizer = MarianTokenizer.from_pretrained(tgt)
    
    return src_tokenizer, src_model, tgt_tokenizer, tgt_model

TRANSLATORS = {
    lang: load_model_pair(lang) for lang in LANGUAGE_MODELS
}

def translate(text, lang_from, lang_to="en"):
    if lang_from == "en":
        return text
    tokenizer, model, _, _ = TRANSLATORS[lang_from]
    inputs = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def back_translate(text, lang_to):
    if lang_to == "en":
        return text
    _, _, tokenizer, model = TRANSLATORS[lang_to]
    inputs = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def load_transcripts():
    chunks = []
    sources = []
    if not os.path.exists(TEXT_FOLDER):
        print("Transcript folder missing!")
        return chunks, sources

    for file in os.listdir(TEXT_FOLDER):
        if file.endswith(".txt"):
            path = os.path.join(TEXT_FOLDER, file)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                cleaned = [line for line in text.split("\n") if line.strip()]
                for line in cleaned:
                    chunks.append(line)
                    sources.append(file)
    return chunks, sources

CHUNKS, SOURCES = load_transcripts()
EMBEDDINGS = embedder.encode(CHUNKS, convert_to_tensor=True)

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def answer_question(query):
    lang = detect_language(query)
    translated = translate(query, lang)

    query_embedding = embedder.encode(translated, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, EMBEDDINGS, top_k=3)[0]

    if not hits or hits[0]['score'] < 0.45:
        response = "❓ I could not find a relevant answer."
    else:
        response = CHUNKS[hits[0]['corpus_id']]

    final_answer = back_translate(response, lang)
    return final_answer, SOURCES[hits[0]['corpus_id']] if hits else "No source"
