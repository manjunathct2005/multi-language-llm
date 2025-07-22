# llm_backend.py

import os
import glob
import torch
import whisper
import numpy as np
import faiss
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator

# Paths
TRANSCRIPT_DIR = "D:/hindupur_dataset/transcripts"
EMBEDDINGS_PATH = "D:/hindupur_dataset/embeddings1.pt"

# Load models
whisper_model = whisper.load_model("base")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Translation functions
def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text

def translate_from_english(text, target_lang):
    try:
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except:
        return text

# Language detection
def detect_language(text):
    try:
        lang = detect(text)
        if lang.startswith("en"):
            return "en"
        elif lang.startswith("hi"):
            return "hi"
        elif lang.startswith("te"):
            return "te"
        else:
            return "unknown"
    except:
        return "unknown"

# Clean transcript text
def clean_text(text):
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if line.strip() and not any(skip in line.lower() for skip in ["thank", "subscribe", "like", "follow"]):
            cleaned.append(line.strip())
    return " ".join(cleaned)

# Load knowledge base and cache embeddings
def knowledge_base(transcript_dir=TRANSCRIPT_DIR):
    if os.path.exists(EMBEDDINGS_PATH):
        data = torch.load(EMBEDDINGS_PATH)
        return data['texts'], data['index'], data['embeddings']

    texts = []
    for file in glob.glob(os.path.join(transcript_dir, "*.txt")):
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
            cleaned = clean_text(raw)
            if cleaned:
                texts.append(cleaned)

    if not texts:
        return [], None, None

    embeddings = embedding_model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    embeddings_np = embeddings.cpu().detach().numpy()
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)

    torch.save({'texts': texts, 'index': index, 'embeddings': embeddings}, EMBEDDINGS_PATH)
    return texts, index, embeddings

# Get answer from KB
def process_input(user_input, texts, index, embeddings, lang="en", style="Summary"):
    if not user_input or index is None or embeddings is None:
        return "❌ No knowledge base available."

    # Step 1: Translate to English
    input_en = translate_to_english(user_input)
    input_emb = embedding_model.encode(input_en, convert_to_tensor=True)

    # Step 2: Search similar
    scores = util.cos_sim(input_emb, embeddings)[0]
    best_idx = int(torch.argmax(scores))
    best_match = texts[best_idx]

    # Step 3: Translate answer back
    if style == "Summary":
        answer = best_match[:500]
    else:
        answer = best_match

    translated_answer = translate_from_english(answer, lang)
    return translated_answer
