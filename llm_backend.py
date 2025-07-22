import os
import torch
import glob
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util

# Paths
TRANSCRIPT_FOLDER = "my1"
EMBEDDINGS_PATH = "D:/hindupur_dataset/embeddings1.pt"

# Load model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load and preprocess text
def load_knowledge_base():
    texts = []
    file_list = glob.glob(os.path.join(TRANSCRIPT_FOLDER, "*.txt"))
    for filepath in file_list:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()
            cleaned = clean_text(raw)
            texts.append(cleaned)
    return texts

# Simple cleaning
def clean_text(text):
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if line and not line.lower().startswith(("part", "#", "what is", "introduction", "page")):
            cleaned.append(line)
    return " ".join(cleaned)

# Embed texts once and save
def get_embeddings(texts):
    if os.path.exists(EMBEDDINGS_PATH):
        data = torch.load(EMBEDDINGS_PATH)
        return data['texts'], data['embeddings']
    embeddings = embed_model.encode(texts, convert_to_tensor=True)
    torch.save({"texts": texts, "embeddings": embeddings}, EMBEDDINGS_PATH)
    return texts, embeddings

# Translate question and answer
def translate(text, src, tgt):
    try:
        return GoogleTranslator(source=src, target=tgt).translate(text)
    except:
        return text

# Detect language (simple trick)
def detect_lang(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text), GoogleTranslator(source='auto', target='en').source
    except:
        return text, 'en'

# Answer function
def answer_question(query):
    query_translated, lang = detect_lang(query)
    texts = load_knowledge_base()
    if not texts:
        return "⚠️ No valid knowledge base found."

    texts, embeddings = get_embeddings(texts)
    query_embedding = embed_model.encode(query_translated, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    best_idx = torch.argmax(scores).item()
    best_match = texts[best_idx]
    final_answer = translate(best_match, src='en', tgt=lang)
    return final_answer
