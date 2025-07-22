import os
import torch
import glob
import numpy as np
from sentence_transformers import SentenceTransformer
from langdetect import detect
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Use local models only
embedding_model = SentenceTransformer("models/sentence-transformers--all-MiniLM-L6-v2")
translator_en = MBartForConditionalGeneration.from_pretrained("models/facebook--mbart-large-50-many-to-many-mmt")
tokenizer_en = MBart50TokenizerFast.from_pretrained("models/facebook--mbart-large-50-many-to-many-mmt")

transcript_folder = "D:/hindupur_dataset/transcripts"
embedding_path = "D:/hindupur_dataset/embeddings.pt"
index_path = "D:/hindupur_dataset/index.npy"
texts_path = "D:/hindupur_dataset/texts.npy"

def translate(text, src_lang, tgt_lang):
    tokenizer_en.src_lang = src_lang
    encoded = tokenizer_en(text, return_tensors="pt")
    generated = translator_en.generate(**encoded, forced_bos_token_id=tokenizer_en.lang_code_to_id[tgt_lang])
    return tokenizer_en.decode(generated[0], skip_special_tokens=True)

def load_transcripts():
    txt_files = glob.glob(os.path.join(transcript_folder, "*.txt"))
    texts = []
    for file in txt_files:
        with open(file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            cleaned = [line.strip() for line in lines if len(line.strip()) > 10]
            texts.extend(cleaned)
    return texts

def build_embeddings(texts):
    embeddings = embedding_model.encode(texts, convert_to_tensor=True)
    torch.save(embeddings, embedding_path)
    np.save(texts_path, np.array(texts))
    return embeddings

def load_embeddings():
    if os.path.exists(embedding_path):
        return torch.load(embedding_path)
    return None

def get_answer(query):
    texts = np.load(texts_path, allow_pickle=True).tolist()
    embeddings = load_embeddings()

    if embeddings is None or len(texts) == 0:
        return "Knowledge base is empty."

    input_lang = detect(query)
    lang_map = {"en": "en_XX", "hi": "hi_IN", "te": "te_IN", "kn": "kn_IN"}

    if input_lang != "en":
        query_en = translate(query, lang_map[input_lang], "en_XX")
    else:
        query_en = query

    query_embedding = embedding_model.encode([query_en], convert_to_tensor=True)
    cosine_scores = torch.nn.functional.cosine_similarity(query_embedding, embeddings)
    top_idx = torch.argmax(cosine_scores).item()
    answer_en = texts[top_idx]

    if input_lang != "en":
        return translate(answer_en, "en_XX", lang_map[input_lang])
    else:
        return answer_en

# Load on start
if not os.path.exists(embedding_path):
    texts = load_transcripts()
    build_embeddings(texts)
