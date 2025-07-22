import os
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
from langdetect import detect

# Use your own folder containing `.txt` files
TEXT_FOLDER = "my1"

# Load and embed all paragraphs
def load_transcripts(folder):
    texts, sources = [], []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            path = os.path.join(folder, filename)
            with open(path, "r", encoding="utf-8") as f:
                paragraphs = f.read().split("\n\n")
                for para in paragraphs:
                    if para.strip():
                        texts.append(para.strip())
                        sources.append(filename)
    return texts, sources

texts, sources = load_transcripts(TEXT_FOLDER)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
text_embeddings = embedder.encode(texts, convert_to_tensor=True)

# Summarization model for answer
qa_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def translate_to_en(text):
    lang = detect(text)
    if lang != "en":
        return GoogleTranslator(source="auto", target="en").translate(text)
    return text

def process_input(query):
    original_lang = detect(query)
    translated_query = translate_to_en(query)

    query_embedding = embedder.encode(translated_query, convert_to_tensor=True)
    similarities = cosine_similarity([query_embedding], text_embeddings)[0]

    top_idx = similarities.argsort()[-3:][::-1]
    top_paragraphs = [texts[i] for i in top_idx]
    top_text = "\n\n".join(top_paragraphs)

    summary = qa_pipeline(top_text, max_length=200, min_length=50, do_sample=False)[0]["summary_text"]

    if original_lang != "en":
        summary = GoogleTranslator(source="en", target=original_lang).translate(summary)

    return f"**Answer:**\n\n{summary}\n\n---\n**Sources:** {', '.join(set([sources[i] for i in top_idx]))}"
