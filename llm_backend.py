import os
import torch
import numpy as np
import whisper
import tempfile
import torchaudio
from sentence_transformers import SentenceTransformer, util
import streamlit as st
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer

# Set directories
TRANSCRIPT_DIR = "D:/hindupur_dataset/transcripts"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
WHISPER_MODEL_NAME = "base"

# Load models once
@st.cache_resource(show_spinner="🔄 Loading models...")
def load_models():
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
    return embedder, whisper_model

embedder, whisper_model = load_models()

# Translation setup
TRANSLATORS = {
    "hi": ("Helsinki-NLP/opus-mt-hi-en", "Helsinki-NLP/opus-mt-en-hi"),
    "te": ("Helsinki-NLP/opus-mt-te-en", "Helsinki-NLP/opus-mt-en-te"),
    "kn": ("Helsinki-NLP/opus-mt-kn-en", "Helsinki-NLP/opus-mt-en-kn"),
}

@st.cache_resource(show_spinner="🔎 Loading translators...")
def load_translators():
    loaded = {}
    for lang, (to_en_model, to_native_model) in TRANSLATORS.items():
        to_en_tokenizer = MarianTokenizer.from_pretrained(to_en_model)
        to_en = MarianMTModel.from_pretrained(to_en_model)

        to_lang_tokenizer = MarianTokenizer.from_pretrained(to_native_model)
        to_lang = MarianMTModel.from_pretrained(to_native_model)

        loaded[lang] = {
            "to_en": (to_en, to_en_tokenizer),
            "from_en": (to_lang, to_lang_tokenizer)
        }
    return loaded

translators = load_translators()

# Clean text utility
def clean_text(text):
    lines = text.split("\n")
    unique_lines = list(dict.fromkeys([l.strip() for l in lines if l.strip()]))
    return " ".join(unique_lines)

# Whisper-compatible mel padding
def pad_or_trim_mel(mel):
    if mel.shape[-1] < 3000:
        pad_len = 3000 - mel.shape[-1]
        mel = torch.nn.functional.pad(mel, (0, pad_len), "constant", 0)
    return mel[:, :3000]

# Transcribe audio using Whisper
def transcribe_audio(path):
    audio = whisper.load_audio(path)
    mel = whisper.log_mel_spectrogram(audio)
    mel = pad_or_trim_mel(mel)
    result = whisper_model.decode(whisper_model.encode(mel))
    return result.text.strip()

# Cache transcript if exists
def get_or_transcribe(audio_path):
    base_name = os.path.basename(audio_path)
    transcript_path = os.path.join(TRANSCRIPT_DIR, base_name + ".txt")

    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            return f.read()

    text = transcribe_audio(audio_path)
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text

# Load knowledge base once
@st.cache_resource(show_spinner="🧠 Loading knowledge base...")
def load_knowledge_base():
    texts, embeddings = [], []
    for fname in os.listdir(TRANSCRIPT_DIR):
        if fname.endswith(".txt"):
            with open(os.path.join(TRANSCRIPT_DIR, fname), "r", encoding="utf-8") as f:
                text = clean_text(f.read())
                texts.append(text)
                embeddings.append(embedder.encode(text, convert_to_tensor=True))
    return texts, embeddings

kb_texts, kb_embeddings = load_knowledge_base()

# Language detection
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# Translate helper
def translate(text, src_lang, direction="to_en"):
    if src_lang not in translators:
        return text

    model, tokenizer = translators[src_lang][direction]
    tokens = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# QA: find best match
def answer_question(question):
    original_lang = detect_language(question)
    q_en = translate(question, original_lang, "to_en") if original_lang != "en" else question

    q_embedding = embedder.encode(q_en, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(q_embedding, kb_embeddings)[0]
    top_idx = torch.argmax(scores).item()
    top_score = scores[top_idx].item()

    if top_score < 0.4:
        return "❌ Sorry, no relevant answer found."

    best_answer = kb_texts[top_idx]
    if original_lang != "en":
        best_answer = translate(best_answer, original_lang, "from_en")
    return best_answer

# Public functions
def process_audio(audio_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir="D:/temp_audio") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name
    return get_or_transcribe(tmp_path)

def process_text_query(query):
    return answer_question(query)
