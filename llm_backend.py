import os
import nltk
from langdetect import detect
from difflib import get_close_matches

# Fix: Ensure Punkt tokenizer is downloaded
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

transcript_folder = "transcripts"

def load_transcripts():
    texts = []
    for file in os.listdir(transcript_folder):
        if file.endswith(".txt"):
            with open(os.path.join(transcript_folder, file), "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts

# Language detection
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def translate_to_english(text, lang):
    return text

def translate_back(text, lang):
    return text

# Main answer logic
def answer_question(query, texts):
    lang = detect_language(query)
    query_en = translate_to_english(query, lang)

    best_match = ""
    best_score = 0

    for text in texts:
        sentences = nltk.sent_tokenize(text.lower())
        matches = get_close_matches(query_en.lower(), sentences, n=1, cutoff=0.4)
        if matches:
            best_match = matches[0]
            best_score = 1

    if best_score > 0:
        return translate_back(best_match, lang)
    else:
        return translate_back("❌ Sorry, no relevant answer found.", lang)

texts = load_transcripts()
