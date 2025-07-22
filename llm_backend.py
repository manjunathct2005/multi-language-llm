import os
import re
import nltk
from langdetect import detect
from difflib import get_close_matches

# Download once
nltk.download("punkt")

# Folder where .txt transcripts are stored
transcript_folder = "my1"

# Clean text: remove repeated lines, symbols, etc.
def clean_text(text):
    # Remove special characters and multiple spaces
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,?!\s]", "", text)
    # Remove repeated lines
    lines = text.split(".")
    seen = set()
    unique_lines = []
    for line in lines:
        line = line.strip()
        if line and line.lower() not in seen:
            seen.add(line.lower())
            unique_lines.append(line)
    return ". ".join(unique_lines)

# Load and clean all transcript text files
def load_transcripts():
    texts = []
    for file in os.listdir(transcript_folder):
        if file.endswith(".txt"):
            with open(os.path.join(transcript_folder, file), "r", encoding="utf-8") as f:
                raw_text = f.read()
                cleaned = clean_text(raw_text)
                texts.append(cleaned)
    return texts

# Very basic language detection
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# Dummy translation passthrough (optional to customize later)
def translate_to_english(text, lang):
    return text

def translate_back(text, lang):
    return text

# Main question answering function
def answer_question(query, texts):
    lang = detect_language(query)
    query_en = translate_to_english(query, lang)

    # Use keyword-based approximate matching
    best_match = ""
    best_score = 0

    for text in texts:
        sentences = nltk.sent_tokenize(text.lower())
        matches = get_close_matches(query_en.lower(), sentences, n=1, cutoff=0.4)
        if matches:
            best_match = matches[0]
            best_score = 1

    if best_score > 0:
        return translate_back(best_match.capitalize(), lang)
    else:
        return translate_back("❌ Sorry, no relevant answer found in the knowledge base.", lang)

# Preload transcript texts
texts = load_transcripts()
