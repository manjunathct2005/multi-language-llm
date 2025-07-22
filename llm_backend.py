
import os
import torch
import re
from sentence_transformers import SentenceTransformer, util
from langdetect import detect
from deep_translator import GoogleTranslator
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# === CONFIG ===
TEXT_FOLDER = r"D:\llm project\my1"
MODEL_NAME = "all-MiniLM-L6-v2"

# === Load Models ===
embedding_model = SentenceTransformer(MODEL_NAME)
mbart_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
mbart_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# === Language Mapping for MBart ===
mbart_lang_map = {
    "en": "en_XX", "hi": "hi_IN", "te": "te_IN", "kn": "kn_IN"
}

def translate_text(text, src_lang, target_lang):
    try:
        return GoogleTranslator(source=src_lang, target=target_lang).translate(text)
    except Exception:
        return text

def mbart_translate(text, src_lang, tgt_lang):
    src_token = mbart_lang_map.get(src_lang, "en_XX")
    tgt_token = mbart_lang_map.get(tgt_lang, "en_XX")
    mbart_tokenizer.src_lang = src_token
    encoded = mbart_tokenizer(text, return_tensors="pt")
    generated_tokens = mbart_model.generate(**encoded, forced_bos_token_id=mbart_tokenizer.lang_code_to_id[tgt_token])
    return mbart_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_transcripts():
    chunks, sources = [], []
    if not os.path.exists(TEXT_FOLDER):
        return [], []
    for file in os.listdir(TEXT_FOLDER):
        if file.endswith(".txt"):
            path = os.path.join(TEXT_FOLDER, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = f.read()
                    sentences = re.split(r'[\n\.]+', raw)
                    for sentence in sentences:
                        sentence = clean_text(sentence)
                        if sentence:
                            chunks.append(sentence)
                            sources.append(file)
            except Exception as e:
                print(f"Error reading {file}: {e}")
    return chunks, sources

def detect_lang(text):
    try:
        return detect(text)
    except:
        return "en"

def embed_chunks(chunks):
    return embedding_model.encode(chunks, convert_to_tensor=True)

# === Load Chunks & Embeddings at Startup ===
CHUNKS, SOURCES = load_transcripts()
EMBEDDINGS = embed_chunks(CHUNKS) if CHUNKS else []

def answer_question(question):
    if not CHUNKS:
        return "❌ No transcript data loaded."
    input_lang = detect_lang(question)
    translated = translate_text(question, input_lang, "en")
    question_embedding = embedding_model.encode(translated, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(question_embedding, EMBEDDINGS)[0]
    best_idx = torch.argmax(scores).item()
    answer_en = CHUNKS[best_idx]
    answer_final = translate_text(answer_en, "en", input_lang)
    return answer_final

def load_available_languages():
    return ["en", "hi", "te", "kn"]
