import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from langdetect import detect
from deep_translator import GoogleTranslator

# Folder where .txt files are stored
TEXT_FOLDER = r"\my1"

# Load sentence transformer model
MODEL = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")

# Preload all sentences and build FAISS index
def load_chunks_and_embeddings():
    chunks = []
    chunk_sources = []

    for file in os.listdir(TEXT_FOLDER):
        if file.endswith(".txt"):
            path = os.path.join(TEXT_FOLDER, file)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                # Split text into small chunks (sentences or paragraphs)
                for para in text.split("\n\n"):
                    clean_para = para.strip()
                    if 20 < len(clean_para) < 1000:  # reasonable size
                        chunks.append(clean_para)
                        chunk_sources.append(file)

    embeddings = MODEL.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    return chunks, chunk_sources, index

CHUNKS, SOURCES, INDEX = load_chunks_and_embeddings()

# Translate to English if not already
def translate_to_english(text):
    lang = detect(text)
    if lang != "en":
        try:
            return GoogleTranslator(source="auto", target="en").translate(text), lang
        except:
            return text, lang
    return text, "en"

# Translate back to original language
def translate_back_to_lang(text, target_lang):
    if target_lang != "en":
        try:
            return GoogleTranslator(source="en", target=target_lang).translate(text)
        except:
            return text
    return text

# Final input processing
def process_input(user_question, top_k=1):
    try:
        translated_qn, original_lang = translate_to_english(user_question)
        question_embedding = MODEL.encode([translated_qn])
        D, I = INDEX.search(np.array(question_embedding).astype("float32"), top_k)

        top_chunk = CHUNKS[I[0][0]].strip()
        source_file = SOURCES[I[0][0]]

        # Translate back if original question was not in English
        final_answer = translate_back_to_lang(top_chunk, original_lang)

        # Add red-colored title using HTML
        html_response = f"<span style='color:red'><b>📌 Most Relevant Answer (from {source_file}):</b></span><br><br>{final_answer}"
        return html_response

    except Exception as e:
        return f"<span style='color:red'>⚠️ Error:</span> {str(e)}"
