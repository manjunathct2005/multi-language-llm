# llm_backend.py
import os
import torch
import numpy as np
import faiss
from langdetect import detect
from googletrans import Translator
from sentence_transformers import SentenceTransformer

# === CONFIG ===
TEXT_FOLDER = r"my1"  # Your actual knowledge folder
MODEL_NAME = "all-MiniLM-L6-v2"

# === LOAD EMBEDDINGS ===
def load_transcripts(text_folder):
    texts = []
    sources = []
    for filename in os.listdir(text_folder):
        if filename.endswith(".txt"):
            filepath = os.path.join(text_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Clean & split into chunks (e.g. sentences)
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                texts.extend(lines)
                sources.extend([filename] * len(lines))
    return texts, sources

# === EMBEDDING ===
model = SentenceTransformer(MODEL_NAME)
texts, sources = load_transcripts(TEXT_FOLDER)
embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(np.array(embeddings))

# === TRANSLATION ===
translator = Translator()

def translate_to_english(text):
    lang = detect(text)
    if lang == 'en':
        return text, lang
    translated = translator.translate(text, dest='en')
    return translated.text, lang

def translate_back(text, lang):
    if lang == 'en':
        return text
    return translator.translate(text, dest=lang).text

# === SEARCH FUNCTION ===
def search_similar_questions(query, top_k=3):
    query_emb = model.encode([query])[0]
    D, I = index.search(np.array([query_emb]), top_k)
    return [(texts[i], sources[i]) for i in I[0]]

# === RESPONSE ===
def process_input(user_query):
    translated_q, user_lang = translate_to_english(user_query)
    results = search_similar_questions(translated_q, top_k=1)

    if not results or not results[0][0]:
        return translate_back("❌ Sorry, I couldn't find anything related. Try rephrasing.", user_lang)

    answer = results[0][0]

    # Format into bullet/code/headings if detected
    formatted = format_output(answer)
    return translate_back(formatted, user_lang)

# === FORMAT ===
def format_output(text):
    import re
    lines = text.strip().split('\n')
    formatted = []
    for line in lines:
        if re.match(r'^\d+[\).]', line) or line.startswith('- '):
            formatted.append(f"- {line}")
        elif line.strip().lower().startswith("step") or ":" in line:
            formatted.append(f"### {line}")
        elif line.strip().startswith("```") or "code" in line.lower():
            formatted.append(f"```python\n{line}\n```")
        else:
            formatted.append(line)
    return "\n".join(formatted)
