import os
import torch
import faiss
import numpy as np
from langdetect import detect
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
import streamlit as st

# CONFIG
TRANSCRIPTS_FOLDER = r"D:\llm project\my1"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource
def load_translator():
    return GoogleTranslator()

@st.cache_resource
def load_available_languages():
    return GoogleTranslator().get_supported_languages()

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def translate_text(text, src_lang, dest_lang):
    if src_lang == dest_lang:
        return text
    try:
        return GoogleTranslator(source=src_lang, target=dest_lang).translate(text)
    except:
        return text

@st.cache_resource
def load_knowledge_base(model):
    texts, embeddings = [], []
    for filename in os.listdir(TRANSCRIPTS_FOLDER):
        if filename.endswith(".txt"):
            path = os.path.join(TRANSCRIPTS_FOLDER, filename)
            with open(path, "r", encoding="utf-8") as file:
                lines = file.readlines()
                for line in lines:
                    sentence = line.strip()
                    if sentence:
                        texts.append(sentence)
                        embedding = model.encode(sentence)
                        embeddings.append(embedding)
    if embeddings:
        dim = len(embeddings[0])
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype("float32"))
        return texts, index, embeddings
    else:
        return [], None, []

def answer_question(question, model, texts, index, embeddings):
    if index is None or not texts:
        return "Knowledge base is empty."

    question_embedding = model.encode(question)
    D, I = index.search(np.array([question_embedding]), k=1)
    if I[0][0] < len(texts):
        return texts[I[0][0]]
    else:
        return "No relevant answer found."
