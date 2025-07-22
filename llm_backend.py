import os
import re
import faiss
import torch
import numpy as np
from langdetect import detect
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util

# === CONFIGURATION ===
TEXT_FOLDER = "my1"  # Path to your transcript text files
MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384  # for MiniLM-based models

# === Load Sentence Transformer Model ===
embedder = SentenceTransformer(MODEL_NAME)

# === Clean text to improve matching ===
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,!?()\s]", "", text)
    return text.strip()

# === Translate text using Deep Translator ===
def translate_text(text, target_lang="en"):
    try:
        translated = GoogleTranslator(source="auto", target=target_lang).translate(text)
        return translated
    except Exception as e:
        print(f"[Translation Error] {e}")
        return text  # Fallback to original if translation fails

# === Load all .txt transcripts and embed ===
def load_knowledge_base():
    documents = []
    embeddings = []

    for filename in os.listdir(TEXT_FOLDER):
        if filename.endswith(".txt"):
            path = os.path.join(TEXT_FOLDER, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = clean_text(f.read())
                    if len(content) > 20:  # Ignore short/incomplete files
                        documents.append(content)
                        emb = embedder.encode(content, convert_to_tensor=True).cpu().numpy()
                        embeddings.append(emb)
            except Exception as e:
                print(f"[Error reading {filename}]: {e}")

    if not embeddings:
        raise ValueError("No valid text files found in the transcript folder.")

    embedding_matrix = np.vstack(embeddings)
    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(embedding_matrix)

    return documents, index, embedding_matrix

# === Load KB on startup ===
print("📥 Loading Knowledge Base...")
DOCUMENTS, INDEX, EMBEDDINGS = load_knowledge_base()
print(f"✅ Loaded {len(DOCUMENTS)} transcript documents.")

# === Process User Input (Text or Transcribed Audio) ===
def process_input(user_input):
    # 1. Detect language
    try:
        input_lang = detect(user_input)
    except:
        input_lang = "en"

    print(f"🌐 Detected language: {input_lang}")

    # 2. Translate to English for semantic search
    translated_input = translate_text(user_input, "en")

    # 3. Embed the query
    query_embedding = embedder.encode(translated_input, convert_to_tensor=True).cpu().numpy()

    # 4. Search using FAISS
    k = 3
    D, I = INDEX.search(np.array([query_embedding]), k)
    best_match_index = I[0][0]

    if D[0][0] > 1.0:
        # No close match
        response_en = "I'm sorry, I couldn't find any relevant information in the knowledge base."
    else:
        response_en = DOCUMENTS[best_match_index]

    # 5. Translate answer back to original language
    final_response = translate_text(response_en, input_lang)
    return final_response
