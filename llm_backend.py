import os
import torch
import glob
import re
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
from langdetect import detect
import warnings

warnings.filterwarnings("ignore")

TRANSCRIPT_DIR = "my1"
EMBEDDINGS_PATH = "embeddings1.pt"
MODEL_PATH = "models/sentence-transformers/all-MiniLM-L6-v2"

# Load embedding model
embedding_model = SentenceTransformer(MODEL_PATH)

def clean_text(text):
    """Remove irrelevant words like part1, symbols like #, *, and extra spaces."""
    text = re.sub(r"(part\s*\d+|#|\*|\-|\s{2,}|`+)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_texts_and_embeddings():
    texts = []
    text_files = glob.glob(os.path.join(TRANSCRIPT_DIR, "*.txt"))
    print(f"🔍 Found {len(text_files)} transcript files")

    for file_path in text_files:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            cleaned = clean_text(text)
            if len(cleaned) > 20:
                texts.append(cleaned)

    if os.path.exists(EMBEDDINGS_PATH):
        print("📦 Loading cached embeddings...")
        embeddings = torch.load(EMBEDDINGS_PATH)
    else:
        print("🧠 Generating embeddings...")
        embeddings = embedding_model.encode(texts, convert_to_tensor=True)
        torch.save(embeddings, EMBEDDINGS_PATH)

    return texts, embeddings

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def translate_to_english(text, source_lang):
    if source_lang == "en":
        return text
    try:
        return GoogleTranslator(source=source_lang, target="en").translate(text)
    except:
        return text

def translate_from_english(text, target_lang):
    if target_lang == "en":
        return text
    try:
        if len(text) < 3 or len(text) > 4999:
            return text
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except:
        return text

def get_answer(question, texts, embeddings):
    question_lang = detect_language(question)
    question_en = translate_to_english(question, question_lang)

    question_embedding = embedding_model.encode(question_en, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, embeddings)[0]

    top_k = min(3, len(texts))
    top_indices = torch.topk(scores, k=top_k).indices
    top_responses = [texts[i] for i in top_indices]

    combined_answer = " ".join(top_responses)
    translated = translate_from_english(combined_answer, question_lang)
    return translated

def process_input(question, texts, embeddings):
    return get_answer(question, texts, embeddings)
