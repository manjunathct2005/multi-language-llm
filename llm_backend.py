# llm_backend.py

import os
import torch
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load default Hugging Face models
device = "cuda" if torch.cuda.is_available() else "cpu"

# Replace with any cloud-friendly model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)

# Translation pipelines
translator_en_hi = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
translator_hi_en = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")
translator_te_en = pipeline("translation", model="Helsinki-NLP/opus-mt-te-en")
translator_en_te = pipeline("translation", model="Helsinki-NLP/opus-mt-en-te")

# Load transcript knowledge base (you must upload .txt files separately)
TRANSCRIPT_FOLDER = "transcripts"
knowledge_base = []

def load_knowledge_base():
    global knowledge_base
    for file in os.listdir(TRANSCRIPT_FOLDER):
        if file.endswith(".txt"):
            path = os.path.join(TRANSCRIPT_FOLDER, file)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                knowledge_base.append((text, embedder.encode(text)))

load_knowledge_base()

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def translate_to_english(text, lang):
    if lang == "hi":
        return translator_hi_en(text)[0]['translation_text']
    elif lang == "te":
        return translator_te_en(text)[0]['translation_text']
    return text

def translate_from_english(text, target_lang):
    if target_lang == "hi":
        return translator_en_hi(text)[0]['translation_text']
    elif target_lang == "te":
        return translator_en_te(text)[0]['translation_text']
    return text

def find_best_answer(query):
    query_embedding = embedder.encode(query)
    best_score = -1
    best_answer = "❌ No relevant answer found in knowledge base."
    for passage, emb in knowledge_base:
        score = util.cos_sim(query_embedding, emb)[0][0]
        if score > best_score:
            best_score = score
            best_answer = passage
    return best_answer

def process_input(user_input):
    lang = detect_language(user_input)
    english_query = translate_to_english(user_input, lang)
    answer_en = find_best_answer(english_query)
    return translate_from_english(answer_en, lang)
