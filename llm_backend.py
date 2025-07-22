import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from langdetect import detect
from deep_translator import GoogleTranslator

# Path where your custom .txt files are stored
TEXT_FOLDER = "my1"

# Load multilingual sentence transformer
EMBEDDINGS_MODEL = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1")

# Load and embed documents
def load_knowledge_base():
    documents = []
    file_names = []
    for filename in os.listdir(TEXT_FOLDER):
        if filename.endswith(".txt"):
            with open(os.path.join(TEXT_FOLDER, filename), "r", encoding="utf-8") as f:
                text = f.read()
                documents.append(text)
                file_names.append(filename)
    embeddings = EMBEDDINGS_MODEL.encode(documents)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    return documents, index, embeddings

DOCUMENTS, INDEX, EMBEDDINGS = load_knowledge_base()

# Detect language and translate to English
def translate_to_english(text):
    lang = detect(text)
    if lang != "en":
        try:
            return GoogleTranslator(source="auto", target="en").translate(text)
        except:
            return text
    return text

# Answer generation logic
def process_input(user_question, top_k=2):
    try:
        # Translate input to English if needed
        question_in_english = translate_to_english(user_question)
        
        # Embed and search
        question_vector = EMBEDDINGS_MODEL.encode([question_in_english])
        D, I = INDEX.search(np.array(question_vector).astype("float32"), top_k)

        # Construct contextual answer
        response = ""
        for idx in I[0]:
            if 0 <= idx < len(DOCUMENTS):
                response += f"**Answer:**\n\n{DOCUMENTS[idx].strip()}\n\n---\n"
        
        if not response:
            return "Sorry, I couldn’t find a relevant answer in your data."
        
        return response
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
