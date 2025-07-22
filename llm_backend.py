import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from langdetect import detect
from deep_translator import GoogleTranslator

TEXT_FOLDER = r"D:\llm project\my1"
MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v1"
model = SentenceTransformer(MODEL_NAME)

# Load all text files as knowledge base
def load_knowledge_base():
    documents = []
    file_names = []
    for filename in os.listdir(TEXT_FOLDER):
        if filename.endswith(".txt"):
            path = os.path.join(TEXT_FOLDER, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    documents.append(content)
                    file_names.append(filename)
    embeddings = model.encode(documents, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    return documents, index, embeddings

DOCUMENTS, INDEX, EMBEDDINGS = load_knowledge_base()

# Translate question to English
def translate_to_english(text):
    lang = detect(text)
    if lang != "en":
        try:
            return GoogleTranslator(source='auto', target='en').translate(text), lang
        except:
            return text, lang
    return text, "en"

# Translate English answer back to original language
def translate_back(text, target_lang):
    if target_lang != "en":
        try:
            return GoogleTranslator(source='en', target=target_lang).translate(text)
        except:
            return text
    return text

# Main function to generate response
def answer_question(user_question, top_k=1):
    try:
        question_en, orig_lang = translate_to_english(user_question)
        question_vector = model.encode([question_en])
        D, I = INDEX.search(np.array(question_vector).astype("float32"), top_k)

        if I[0][0] < len(DOCUMENTS):
            answer = DOCUMENTS[I[0][0]].strip()
        else:
            return translate_back("Sorry, no relevant answer found in your data.", orig_lang)

        final_answer = translate_back(answer, orig_lang)
        return f"🔍 **Answer**:\n\n<span style='color:red'>{final_answer}</span>", orig_lang

    except Exception as e:
        return f"⚠️ Error: {str(e)}", "en"
