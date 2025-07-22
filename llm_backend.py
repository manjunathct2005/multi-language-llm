import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from langdetect import detect
from deep_translator import GoogleTranslator

# === CONFIG ===
TEXT_FOLDER = "my1"  # Update if needed
EMBEDDINGS_MODEL = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")

# === Load Texts and Embeddings ===
def load_knowledge_base():
    documents = []
    file_names = []
    for filename in os.listdir(TEXT_FOLDER):
        if filename.endswith(".txt"):
            with open(os.path.join(TEXT_FOLDER, filename), "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:  # Skip empty files
                    documents.append(text)
                    file_names.append(filename)
    embeddings = EMBEDDINGS_MODEL.encode(documents, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    return documents, index, embeddings

DOCUMENTS, INDEX, EMBEDDINGS = load_knowledge_base()

# === Translation ===
def translate_to_english(text):
    try:
        lang = detect(text)
        if lang != "en":
            return GoogleTranslator(source="auto", target="en").translate(text), lang
        return text, "en"
    except:
        return text, "en"

def translate_back(text, target_lang):
    try:
        if target_lang != "en":
            return GoogleTranslator(source="en", target=target_lang).translate(text)
        return text
    except:
        return text

# === Main Q&A Processing ===
def process_input(user_question, top_k=1):
    try:
        # Translate question to English
        question_en, detected_lang = translate_to_english(user_question)

        # Embed and search
        question_vector = EMBEDDINGS_MODEL.encode([question_en])
        D, I = INDEX.search(np.array(question_vector).astype("float32"), top_k)

        # Get the most relevant answer
        response = ""
        for idx in I[0]:
            if 0 <= idx < len(DOCUMENTS):
                best_answer = DOCUMENTS[idx].strip()
                # Highlight using red color in Markdown
                response = f"<span style='color:red'><b>{best_answer}</b></span>"
                break

        if not response:
            fallback = "Sorry, I couldn’t find a relevant answer in your documents."
            return translate_back(fallback, detected_lang)

        return translate_back(response, detected_lang)

    except Exception as e:
        return f"⚠️ Error: {str(e)}"
