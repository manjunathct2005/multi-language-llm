import os
import glob
import torch
import whisper
import numpy as np
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
import faiss

# Paths
TRANSCRIPT_DIR = "D:/hindupur_dataset/transcripts"
EMBEDDINGS_PATH = "D:/hindupur_dataset/embeddings1.pt"

# Load models
whisper_model = whisper.load_model("base")  # change to "small" if needed
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Translate text to English (for retrieval) and back to original language
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

# Load or build knowledge base
def clean_text(text):
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if line and not line.lower().startswith(("background noise", "inaudible")):
            cleaned.append(line)
    return "\n".join(cleaned)

def load_knowledge_base():
    if not os.path.exists(TRANSCRIPT_DIR):
        os.makedirs(TRANSCRIPT_DIR)

    files = glob.glob(os.path.join(TRANSCRIPT_DIR, "*.txt"))
    knowledge_blocks = []
    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if text:
                cleaned = clean_text(text)
                if cleaned:
                    knowledge_blocks.append(cleaned)

    if not knowledge_blocks:
        return [], None, None

    # Generate embeddings
    embeddings = embedding_model.encode(knowledge_blocks, convert_to_tensor=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.cpu().detach().numpy())

    return knowledge_blocks, index, embeddings

knowledge_base, faiss_index, kb_embeddings = load_knowledge_base()

# Main processor
def process_input(query):
    if not query.strip():
        return "Query is empty.", 0.0

    try:
        original_lang = detect(query)
    except:
        return "Only Telugu/English questions are supported.", 0.0

    if original_lang not in ['en', 'te', 'hi']:
        return "Only Telugu/English/Hindi questions are supported.", 0.0

    translated_query = translate_to_english(query)
    query_embedding = embedding_model.encode(translated_query, convert_to_tensor=True)
    query_embedding_np = query_embedding.cpu().detach().numpy()

    D, I = faiss_index.search(np.array([query_embedding_np]), k=1)
    best_match_idx = I[0][0]
    score = float(D[0][0])

    if score > 1.5:  # Adjust threshold based on dataset size
        return "⚠️ No relevant answer found in your knowledge base.", score

    matched_text = knowledge_base[best_match_idx]
    answer_translated = translate_from_english(matched_text, original_lang)
    return answer_translated, score
