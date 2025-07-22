import os
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from langdetect import detect

# Load sentence transformer from Hugging Face hub
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load translation models (from Hugging Face)
translator_en_hi = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
translator_hi_en = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")
translator_multi = pipeline("translation", model="facebook/mbart-large-50-many-to-many-mmt")

# Cache to avoid reloading
transcript_folder = "transcripts"
embedding_cache_path = "embeddings.pt"

def load_transcripts():
    texts = []
    for fname in os.listdir(transcript_folder):
        if fname.endswith(".txt"):
            with open(os.path.join(transcript_folder, fname), "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts

def get_embeddings(texts):
    if os.path.exists(embedding_cache_path):
        return torch.load(embedding_cache_path)
    embeddings = embedder.encode(texts, convert_to_tensor=True)
    torch.save(embeddings, embedding_cache_path)
    return embeddings

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def translate_to_english(text, lang):
    if lang == "hi":
        return translator_hi_en(text)[0]["translation_text"]
    elif lang in ["te", "kn"]:
        return translator_multi(text, src_lang=lang, tgt_lang="en")[0]["translation_text"]
    return text

def translate_back(text, lang):
    if lang == "hi":
        return translator_en_hi(text)[0]["translation_text"]
    elif lang in ["te", "kn"]:
        return translator_multi(text, src_lang="en", tgt_lang=lang)[0]["translation_text"]
    return text

def answer_question(query, texts, embeddings):
    lang = detect_language(query)
    query_en = translate_to_english(query, lang)
    query_embedding = embedder.encode(query_en, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, embeddings)[0]
    best_idx = torch.argmax(scores).item()
    best_answer = texts[best_idx]
    return translate_back(best_answer, lang)

# Initialization
texts = load_transcripts()
embeddings = get_embeddings(texts)
