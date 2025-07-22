import os
import re
import torch
import numpy as np
import torchaudio
from langdetect import detect
from transformers import pipeline, MarianMTModel, MarianTokenizer
from sentence_transformers import SentenceTransformer, util
import whisper
from tqdm import tqdm

# === CONFIG ===
TEXT_FOLDER = r"D:\llm project\my1"
EMBEDDING_MODEL_PATH = r"models/models--sentence-transformers--all-MiniLM-L6-v2"
WHISPER_MODEL_PATH = r"models/models--openai--whisper-base"
TRANSLATOR_HI_EN_PATH = r"models/models--Helsinki-NLP--opus-mt-hi-en"
TRANSLATOR_EN_HI_PATH = r"models/models--Helsinki-NLP--opus-mt-en-hi"
TEMP_AUDIO_DIR = r"D:\temp_chunks"

os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# === LOAD MODELS ===
device = "cuda" if torch.cuda.is_available() else "cpu"

embedder = SentenceTransformer(EMBEDDING_MODEL_PATH, device=device)
whisper_model = whisper.load_model(WHISPER_MODEL_PATH, device=device)

hi_en_tokenizer = MarianTokenizer.from_pretrained(TRANSLATOR_HI_EN_PATH)
hi_en_model = MarianMTModel.from_pretrained(TRANSLATOR_HI_EN_PATH).to(device)

en_hi_tokenizer = MarianTokenizer.from_pretrained(TRANSLATOR_EN_HI_PATH)
en_hi_model = MarianMTModel.from_pretrained(TRANSLATOR_EN_HI_PATH).to(device)

# === UTILS ===
def translate(text, src_lang, tgt_lang):
    if src_lang == tgt_lang:
        return text
    if src_lang == "hi" and tgt_lang == "en":
        tokenizer, model = hi_en_tokenizer, hi_en_model
    elif src_lang == "en" and tgt_lang == "hi":
        tokenizer, model = en_hi_tokenizer, en_hi_model
    else:
        return text  # not supported
    batch = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    generated = model.generate(**batch)
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"(?:\b\w+\b\s+){0,2}\1", "", text)
    return text.strip()

# === AUDIO TRANSCRIPTION ===
def pad_or_trim_mel(mel):
    if mel.shape[-1] < 3000:
        pad_width = 3000 - mel.shape[-1]
        mel = torch.nn.functional.pad(mel, (0, pad_width), mode='constant', value=0)
    else:
        mel = mel[:, :3000]
    return mel

def transcribe_audio_file(file_path, save_path):
    audio = whisper.load_audio(file_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    mel = pad_or_trim_mel(mel)
    options = whisper.DecodingOptions(fp16=False)
    result = whisper.decode(whisper_model, mel, options)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(result.text)
    return result.text

def batch_transcribe_audio(folder):
    for file in tqdm(os.listdir(folder)):
        if file.endswith(".wav"):
            base = os.path.splitext(file)[0]
            txt_path = os.path.join(TEXT_FOLDER, f"{base}.txt")
            if not os.path.exists(txt_path):
                try:
                    audio_path = os.path.join(folder, file)
                    transcribe_audio_file(audio_path, txt_path)
                except Exception as e:
                    print(f"❌ Failed: {file} - {e}")

# === KNOWLEDGE BASE ===
def load_knowledge_embeddings(folder):
    texts, embeddings = [], []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                raw = f.read()
            cleaned = clean_text(raw)
            texts.append(cleaned)
            emb = embedder.encode(cleaned, convert_to_tensor=True)
            embeddings.append(emb)
    return texts, torch.stack(embeddings)

# === ANSWER SYSTEM ===
knowledge_texts, knowledge_embeddings = load_knowledge_embeddings(TEXT_FOLDER)

def answer_question(user_input):
    input_lang = detect(user_input)
    question_en = translate(user_input, input_lang, "en")

    query_embedding = embedder.encode(question_en, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, knowledge_embeddings)[0]
    best_idx = torch.argmax(scores).item()
    best_score = scores[best_idx].item()

    if best_score < 0.4:
        return translate("Sorry, I couldn't find the answer. Try searching online.", "en", input_lang)

    answer_en = knowledge_texts[best_idx]
    return translate(answer_en, "en", input_lang)
