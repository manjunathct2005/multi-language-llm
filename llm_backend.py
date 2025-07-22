import os
import re
import faiss
import numpy as np
from langdetect import detect
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

# === CONFIG ===
TEXT_FOLDER = "my1"  # 📁 Place your .txt files in ./docs folder (same dir as app)
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# === Clean & Chunk ===
def clean_and_chunk(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove emojis/non-ASCII
    text = re.sub(r"\s{2,}", " ", text)
    raw_chunks = re.split(r"\n\s*---\s*\n", text)
    return [chunk.strip() for chunk in raw_chunks if len(chunk.strip()) > 50]

# === Load & Embed ===
def load_documents():
    chunks = []
    for file in os.listdir(TEXT_FOLDER):
        if file.endswith(".txt"):
            with open(os.path.join(TEXT_FOLDER, file), "r", encoding="utf-8") as f:
                chunks += clean_and_chunk(f.read())
    return chunks

@st.cache_resource
def build_knowledge_base():
    chunks = load_documents()
    embeddings = model.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return chunks, index, embeddings.shape[1]

knowledge_base, faiss_index, dim = build_knowledge_base()

# === Language Handling ===
def detect_lang(text):
    try:
        return detect(text)
    except:
        return "en"

def to_english(text):
    lang = detect_lang(text)
    if lang != "en":
        try:
            return GoogleTranslator(source=lang, target="en").translate(text), lang
        except:
            return text, "en"
    return text, "en"

def from_english(text, target_lang):
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except:
        return text

# === Main Logic ===
def answer_question(query, original_lang):
    query_en, detected_lang = to_english(query)
    query_vec = model.encode([query_en], convert_to_numpy=True)
    D, I = faiss_index.search(query_vec, k=1)

    if I[0][0] < len(knowledge_base):
        result = knowledge_base[I[0][0]]
        final_answer = from_english(result, original_lang)
        return final_answer
    else:
        return "Sorry, no relevant answer found."

def load_available_languages():
    return ["English", "Telugu"]
