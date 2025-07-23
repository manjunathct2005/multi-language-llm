import os
import re
import torch
import numpy as np
from langdetect import detect
from googletrans import Translator
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Configuration
TEXT_FOLDER = "my1"
MODEL_NAME = "all-MiniLM-L6-v2"

# Load translation and embedding model
translator = Translator()
model = SentenceTransformer(MODEL_NAME)

# Text cleaning and chunking
def clean_and_chunk(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"---+", "\n---\n", text)
    text = re.sub(r"\n{2,}", "\n", text)

    raw_chunks = re.split(r"\n\s*---\s*\n", text)
    clean_chunks = []

    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk or len(chunk) < 50:
            continue
        lines = chunk.split("\n")
        cleaned = "\n".join([line.strip() for line in lines if line.strip()])
        clean_chunks.append(cleaned)
    
    return clean_chunks
