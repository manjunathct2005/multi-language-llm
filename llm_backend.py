import os
import glob
import torch
import faiss
import re
from langdetect import detect
from sentence_transformers import SentenceTransformer, util
from deep_translator import DeeplTranslator  # Replacing GoogleTranslator

# === Configuration ===
TRANSCRIPT_DIR = "my1"  # folder containing .txt files
EMBEDDINGS_PATH = r"D:\hindupur_dataset\embeddings1.pt"
MODEL_NAME = "all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

embedding_model = SentenceTransformer(MODEL_NAME, device=DEVICE)

# === Load & Clean Text Files ===
def load_transcripts():
    texts, filenames = [], []
    for filepath in glob.glob(os.path.join(TRANSCRIPT_DIR, "*.txt")):
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            text = re.sub(r"(part\s*\d+|what is\s*data\s*science.*?:?)", "", text, flags=re.I)
            text = re.sub(r"[^a-zA-Z0-9\s.,]", "", text)  # Remove unwanted characters
            text = re.sub(r"\s+", " ", text)  # Normalize spacing
            texts.append(text.strip())
            filenames.append(os.path.basename(filepath))
    return texts, filenames

# === Embedding Loader/Saver ===
def get_embeddings():
    if os.path.exists(EMBEDDINGS_PATH):
        return torch.load(EMBEDDINGS_PATH)
    texts, _ = load_transcripts()
    embeddings = embedding_model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    torch.save(embeddings, EMBEDDINGS_PATH)
    return embeddings

# === Translation Functions ===
def translate_to_english(text):
    try:
        return DeeplTranslator(source="auto", target="en").translate(text)
    except:
        return text

def translate_from_english(text, target_lang):
    if target_lang.lower().startswith("en"):
        return text
    try:
        return DeeplTranslator(source="en", target=target_lang).translate(text)
    except:
        return text

# === Answer a Question ===
def answer_question(question_or_audio):
    if isinstance(question_or_audio, str):
        return process_question(question_or_audio)
    else:
        return "Audio input currently not supported in this backend version."

def process_question(question):
    texts, filenames = load_transcripts()
    embeddings = get_embeddings()

    original_lang = detect(question)
    translated_q = translate_to_english(question)

    q_embedding = embedding_model.encode(translated_q, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(q_embedding, embeddings)[0]

    best_idx = torch.argmax(scores).item()
    answer_raw = texts[best_idx]
    answer = answer_raw.strip()[:1000]  # limit response size

    translated_ans = translate_from_english(answer, original_lang)
    return translated_ans

# === Load Available Languages ===
def load_available_languages():
    return ["en", "hi", "te", "kn", "ta", "ml", "ur"]
