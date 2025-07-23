import os
import re
import torch
from deep_translator import GoogleTranslator
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Clean up config
TRANSCRIPTS_DIR = "my1"
MODEL_NAME = "all-MiniLM-L6-v2"  # Used directly from HuggingFace cache
MAX_INPUT_LENGTH = 5000  # Safe limit for deep_translator

# Load model once
embedding_model = SentenceTransformer(MODEL_NAME)

def clean_text(text):
    # Remove headings, part numbers, symbols, multiple spaces, etc.
    text = re.sub(r"(part\s*\d+|chapter\s*\d+|section\s*\d+)", "", text, flags=re.I)
    text = re.sub(r"[#*●•■◆►▪️➤➡️]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_texts_and_embeddings():
    all_texts = []
    all_embeddings = []

    for filename in os.listdir(TRANSCRIPTS_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(TRANSCRIPTS_DIR, filename), "r", encoding="utf-8") as f:
                raw_text = f.read()
                cleaned_text = clean_text(raw_text)
                if cleaned_text.strip():
                    all_texts.append(cleaned_text)
                    embedding = embedding_model.encode([cleaned_text], convert_to_tensor=True)
                    all_embeddings.append(embedding)

    if not all_texts:
        raise FileNotFoundError("No valid transcripts found in the directory.")

    all_embeddings = torch.cat(all_embeddings, dim=0)
    return all_texts, all_embeddings

def detect_language(text):
    try:
        return GoogleTranslator().detect(text)
    except Exception:
        return "en"

def translate_to_english(text):
    lang = detect_language(text)
    if lang != "en":
        try:
            return GoogleTranslator(source=lang, target="en").translate(text), lang
        except Exception:
            return text, lang
    return text, lang

def translate_from_english(text, target_lang):
    if target_lang != "en" and text.strip():
        try:
            if len(text) > MAX_INPUT_LENGTH:
                text = text[:MAX_INPUT_LENGTH]
            return GoogleTranslator(source="en", target=target_lang).translate(text)
        except Exception:
            return text
    return text

def get_answer(question, texts, embeddings):
    question_embedding = embedding_model.encode([question], convert_to_tensor=True)
    sims = cosine_similarity(question_embedding.cpu().numpy(), embeddings.cpu().numpy())[0]
    top_idx = sims.argsort()[::-1][:3]  # Top 3 relevant answers
    top_answers = [texts[i] for i in top_idx]
    return " ".join(top_answers)

def process_input(user_question, texts, embeddings):
    question_in_english, original_lang = translate_to_english(user_question)
    answer_english = get_answer(question_in_english, texts, embeddings)
    final_answer = translate_from_english(answer_english, original_lang)
    return final_answer
