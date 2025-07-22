import os
import torch
import re
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from sentence_transformers import SentenceTransformer, util
from langdetect import detect
import glob

# Load translation models
en2hi_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
en2hi_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Folder path
TRANSCRIPT_FOLDER = "D:/hindupur_dataset/transcripts"

# Clean text function
def clean_text(text):
    text = re.sub(r"#.*", "", text)                     # remove markdown headers
    text = re.sub(r"\bpart\s*\d+", "", text, flags=re.I) # remove 'part 1', 'Part 2'
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)       # keep only alphanumeric + punctuation
    text = re.sub(r"\s+", " ", text).strip()             # remove extra whitespace
    return text

# Load cleaned transcript data
def load_transcripts():
    sentences = []
    for file in glob.glob(os.path.join(TRANSCRIPT_FOLDER, "*.txt")):
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                cleaned = clean_text(line)
                if len(cleaned.split()) > 4:  # skip too short lines
                    sentences.append(cleaned)
    embeddings = embedding_model.encode(sentences, convert_to_tensor=True)
    return sentences, embeddings

# Translate to English (if needed)
def translate_to_english(text, lang):
    if lang == "en":
        return text
    en2hi_tokenizer.src_lang = lang
    encoded = en2hi_tokenizer(text, return_tensors="pt")
    generated = en2hi_model.generate(**encoded, forced_bos_token_id=en2hi_tokenizer.lang_code_to_id["en_XX"])
    return en2hi_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

# Translate back to original language
def translate_to_original(text, lang):
    if lang == "en":
        return text
    en2hi_tokenizer.src_lang = "en_XX"
    encoded = en2hi_tokenizer(text, return_tensors="pt")
    generated = en2hi_model.generate(**encoded, forced_bos_token_id=en2hi_tokenizer.lang_code_to_id[f"{lang}_XX"])
    return en2hi_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

# Answer generator
def answer_question(question, sentences, embeddings):
    detected_lang = detect(question)
    question_in_english = translate_to_english(question, detected_lang)
    question_embedding = embedding_model.encode(question_in_english, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, embeddings)[0]
    top_idx = torch.argmax(scores).item()
    best_answer = sentences[top_idx]
    return translate_to_original(best_answer, detected_lang)
import os
import torch
import re
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from sentence_transformers import SentenceTransformer, util
from langdetect import detect
import glob

# Load translation models
en2hi_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
en2hi_model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Folder path
TRANSCRIPT_FOLDER = "D:/hindupur_dataset/transcripts"

# Clean text function
def clean_text(text):
    text = re.sub(r"#.*", "", text)                     # remove markdown headers
    text = re.sub(r"\bpart\s*\d+", "", text, flags=re.I) # remove 'part 1', 'Part 2'
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)       # keep only alphanumeric + punctuation
    text = re.sub(r"\s+", " ", text).strip()             # remove extra whitespace
    return text

# Load cleaned transcript data
def load_transcripts():
    sentences = []
    for file in glob.glob(os.path.join(TRANSCRIPT_FOLDER, "*.txt")):
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                cleaned = clean_text(line)
                if len(cleaned.split()) > 4:  # skip too short lines
                    sentences.append(cleaned)
    embeddings = embedding_model.encode(sentences, convert_to_tensor=True)
    return sentences, embeddings

# Translate to English (if needed)
def translate_to_english(text, lang):
    if lang == "en":
        return text
    en2hi_tokenizer.src_lang = lang
    encoded = en2hi_tokenizer(text, return_tensors="pt")
    generated = en2hi_model.generate(**encoded, forced_bos_token_id=en2hi_tokenizer.lang_code_to_id["en_XX"])
    return en2hi_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

# Translate back to original language
def translate_to_original(text, lang):
    if lang == "en":
        return text
    en2hi_tokenizer.src_lang = "en_XX"
    encoded = en2hi_tokenizer(text, return_tensors="pt")
    generated = en2hi_model.generate(**encoded, forced_bos_token_id=en2hi_tokenizer.lang_code_to_id[f"{lang}_XX"])
    return en2hi_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

# Answer generator
def answer_question(question, sentences, embeddings):
    detected_lang = detect(question)
    question_in_english = translate_to_english(question, detected_lang)
    question_embedding = embedding_model.encode(question_in_english, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, embeddings)[0]
    top_idx = torch.argmax(scores).item()
    best_answer = sentences[top_idx]
    return translate_to_original(best_answer, detected_lang)
