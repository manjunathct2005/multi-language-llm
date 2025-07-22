import os
import re
import faiss
import torch
import numpy as np
from langdetect import detect
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer

TEXT_FOLDER = "my1"
MODEL_NAME = "all-MiniLM-L6-v2"

# Load model
model = SentenceTransformer(MODEL_NAME)

# Load & clean text
def load_documents(folder):
    docs = {}
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                content = re.sub(r'\s+', ' ', content)
                docs[filename] = content
    return docs

# Embed documents
def build_embeddings(docs):
    sentences = list(docs.values())
    embeddings = model.encode(sentences, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, sentences, list(docs.keys())

# Translate text
def translate(text, src, tgt):
    if src == tgt:
        return text
    return GoogleTranslator(source=src, target=tgt).translate(text)

# Load knowledge base
documents = load_documents(TEXT_FOLDER)
index, sentences, filenames = build_embeddings(documents)

# Process user input
def process_input(user_input):
    lang = detect(user_input)
    question_en = translate(user_input, lang, "en")
    question_embedding = model.encode([question_en])
    
    D, I = index.search(np.array(question_embedding), k=1)
    if D[0][0] > 1.0:
        return translate("Sorry, I couldn't find a relevant answer.", "en", lang)
    
    answer_en = sentences[I[0][0]]
    return translate(answer_en, "en", lang)
