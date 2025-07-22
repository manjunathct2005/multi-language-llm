import os
import re
import torch
import numpy as np
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
from argostranslate.package import install_from_path, get_available_packages
from argostranslate.translate import load_installed_packages, translate

# === PATH CONFIG ===
TEXT_FOLDER = r"D:\llm project\my1"
EMBEDDING_DIM = 384
model = SentenceTransformer("all-MiniLM-L6-v2")

# === TRANSLATION SETUP ===
load_installed_packages()

def install_lang_package(from_code, to_code):
    packages = get_available_packages()
    for pkg in packages:
        if pkg.from_code == from_code and pkg.to_code == to_code:
            install_from_path(pkg.download())

# === Clean, Embed and Load Knowledge Base ===
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_knowledge_base():
    texts = []
    embeddings = []
    for fname in os.listdir(TEXT_FOLDER):
        if fname.endswith(".txt"):
            path = os.path.join(TEXT_FOLDER, fname)
            with open(path, "r", encoding="utf-8") as f:
                content = clean_text(f.read())
                if content:
                    texts.append(content)
                    embeddings.append(model.encode(content, convert_to_tensor=True))
    return texts, embeddings

kb_texts, kb_embeddings = load_knowledge_base()

# === Translator ===
def translate_to_english(text, lang):
    try:
        return translate(text, lang, "en")
    except:
        return text

def translate_from_english(text, lang):
    try:
        return translate(text, "en", lang)
    except:
        return text

# === Main Answer Function ===
def get_answer(query):
    detected_lang = detect(query)
    query_en = translate_to_english(query, detected_lang)

    query_embedding = model.encode(query_en, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, kb_embeddings)[0]
    top_idx = torch.argmax(scores).item()

    if scores[top_idx] < 0.3:
        return translate_from_english("Sorry, I couldn’t find a good answer for your question.", detected_lang)

    response = kb_texts[top_idx]
    return translate_from_english(response, detected_lang)
