import os
import torch
from sentence_transformers import SentenceTransformer, util
from langdetect import detect
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import re

# Download models from Hugging Face (NOT from local)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
model_hi = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer_hi = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

device = "cuda" if torch.cuda.is_available() else "cpu"
model_hi = model_hi.to(device)

# Language code map
lang_code = {
    "hi": "hi_IN",
    "en": "en_XX",
    "te": "te_IN",
    "kn": "kn_IN"
}

# Clean transcript text
def clean_text(text):
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"(?i)(this (was|is|has been)|thank you|thanks|welcome|okay).*", "", text)
    return text.strip()

# Load knowledge base from transcripts
def load_transcripts(folder_path="transcripts"):
    texts = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                content = f.read()
                texts.append(clean_text(content))
    embeddings = embedder.encode(texts, convert_to_tensor=True)
    return texts, embeddings

# Translate text to English for search
def translate_to_english(text):
    lang = detect(text)
    if lang == "en":
        return text, "en"
    tokenizer_hi.src_lang = lang_code.get(lang, "en_XX")
    encoded = tokenizer_hi(text, return_tensors="pt").to(device)
    generated = model_hi.generate(**encoded, forced_bos_token_id=tokenizer_hi.lang_code_to_id["en_XX"])
    translated = tokenizer_hi.batch_decode(generated, skip_special_tokens=True)[0]
    return translated, lang

# Translate answer back to original language
def translate_to_original(text, target_lang):
    if target_lang == "en":
        return text
    tokenizer_hi.src_lang = "en_XX"
    encoded = tokenizer_hi(text, return_tensors="pt").to(device)
    generated = model_hi.generate(**encoded, forced_bos_token_id=tokenizer_hi.lang_code_to_id[lang_code.get(target_lang, "en_XX")])
    translated = tokenizer_hi.batch_decode(generated, skip_special_tokens=True)[0]
    return translated

# Main answer function
def get_answer(user_question, texts, embeddings):
    question_en, orig_lang = translate_to_english(user_question)
    question_emb = embedder.encode(question_en, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(question_emb, embeddings)[0]
    best_idx = torch.argmax(scores).item()
    answer = texts[best_idx]
    translated_answer = translate_to_original(answer, orig_lang)
    return translated_answer
