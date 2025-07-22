import os
import torch
import whisper
import glob
from langdetect import detect
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator

# Load models
def load_models():
    whisper_model = whisper.load_model("base")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    mbart_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    return whisper_model, embedder, mbart_model, mbart_tokenizer

whisper_model, embedder, mbart_model, mbart_tokenizer = load_models()

# Language detection
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# Translation
def translate_text(text, src_lang, tgt_lang):
    try:
        return GoogleTranslator(source=src_lang, target=tgt_lang).translate(text)
    except Exception as e:
        print("Translation failed:", e)
        return text

# Text cleaning
def clean_text(text):
    lines = text.splitlines()
    cleaned = [
        line.strip() for line in lines
        if len(line.strip()) > 5
        and not line.strip().lower().startswith(("chapter", "part", "*", "#", "index"))
    ]
    return " ".join(cleaned)

# Load knowledge base
def load_knowledge_base(transcript_folder="my1"):
    texts, embeddings = [], []
    for file in glob.glob(os.path.join(transcript_folder, "*.txt")):
        with open(file, "r", encoding="utf-8") as f:
            content = f.read()
            cleaned = clean_text(content)
            if cleaned:
                texts.append(cleaned)
                embeddings.append(embedder.encode(cleaned, convert_to_tensor=True))
    return texts, embeddings

# Get best-matching answer
def get_answer(query, texts, embeddings):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    best_score = torch.max(scores).item()
    best_index = torch.argmax(scores).item()
    return texts[best_index] if best_score > 0.4 else None
