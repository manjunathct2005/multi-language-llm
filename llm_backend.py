import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langdetect import detect
from deep_translator import DeeplTranslator, single_detection
import re

# Use relative path for deployment
TEXT_FOLDER = "my1"

EMBEDDINGS_MODEL = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")

def load_knowledge_base():
    documents = []
    sources = []
    if not os.path.exists(TEXT_FOLDER):
        raise FileNotFoundError(f"Text folder not found: {TEXT_FOLDER}")
    for filename in os.listdir(TEXT_FOLDER):
        if filename.endswith(".txt"):
            with open(os.path.join(TEXT_FOLDER, filename), "r", encoding="utf-8") as f:
                text = f.read()
                chunks = split_into_chunks(text)
                documents.extend(chunks)
                sources.extend([filename] * len(chunks))
    embeddings = EMBEDDINGS_MODEL.encode(documents)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    return documents, sources, index

def split_into_chunks(text, chunk_size=500):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def translate_to_english(text):
    lang = detect(text)
    if lang != "en":
        try:
            return DeeplTranslator(source="auto", target="en").translate(text)
        except:
            return text
    return text

CHUNKS, SOURCES, INDEX = load_knowledge_base()

def answer_question(question, top_k=3):
    try:
        translated_question = translate_to_english(question)
        question_vector = EMBEDDINGS_MODEL.encode([translated_question])
        D, I = INDEX.search(np.array(question_vector).astype("float32"), top_k)

        results = []
        for idx in I[0]:
            if 0 <= idx < len(CHUNKS):
                results.append(f"<span style='color:red'><b>Answer:</b></span><br>{CHUNKS[idx]}")
        
        if results:
            return "<br><hr><br>".join(results)
        else:
            return "<span style='color:red'><b>Sorry:</b></span> No relevant answer found."
    except Exception as e:
        return f"<span style='color:red'>⚠️ Error:</span> {str(e)}"
