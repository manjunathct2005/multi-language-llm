import os
import torch
import faiss
import re
from langdetect import detect
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from sentence_transformers import SentenceTransformer, util

# Load multilingual models
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
translate_en = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer_en = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# Set to use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
translate_en = translate_en.to(device)

# Directory to load text files from
TRANSCRIPT_DIR = "transcripts"
knowledge_base = []
embedding_dim = 384  # for MiniLM-L6-v2

# Load all transcript files
def load_knowledge():
    texts = []
    for file in os.listdir(TRANSCRIPT_DIR):
        if file.endswith(".txt"):
            with open(os.path.join(TRANSCRIPT_DIR, file), "r", encoding="utf-8") as f:
                raw = f.read()
                # Clean repetitive/irrelevant lines
                lines = [line.strip() for line in raw.splitlines() if len(line.strip()) > 20]
                text = "\n".join(sorted(set(lines), key=lines.index))
                texts.append(text)
    return texts

# Translate non-English queries to English
def translate_to_en(text):
    src_lang = detect(text)
    if src_lang == 'en':
        return text, 'en'
    tokenizer_en.src_lang = f"{src_lang}_XX"
    encoded = tokenizer_en(text, return_tensors="pt", truncation=True).to(device)
    generated = translate_en.generate(**encoded, forced_bos_token_id=tokenizer_en.lang_code_to_id["en_XX"])
    translated = tokenizer_en.decode(generated[0], skip_special_tokens=True)
    return translated, src_lang

# Translate answers back to user’s original language
def translate_from_en(text, tgt_lang):
    if tgt_lang == 'en':
        return text
    tokenizer_en.src_lang = "en_XX"
    encoded = tokenizer_en(text, return_tensors="pt", truncation=True).to(device)
    generated = translate_en.generate(**encoded, forced_bos_token_id=tokenizer_en.lang_code_to_id[f"{tgt_lang}_XX"])
    translated = tokenizer_en.decode(generated[0], skip_special_tokens=True)
    return translated

# Embed and index all sentences
def build_vector_store(texts):
    sentences = []
    file_index = []
    for doc in texts:
        for sent in re.split(r'(?<=[.?!])\s+', doc):
            if 30 < len(sent) < 1000:
                sentences.append(sent)
                file_index.append(sent)
    embeddings = embedding_model.encode(sentences, convert_to_tensor=False)
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    return index, sentences

# Run similarity search and return best result
def process_input(query):
    global knowledge_base, kb_index

    translated_query, input_lang = translate_to_en(query)
    query_embedding = embedding_model.encode([translated_query])
    D, I = kb_index.search(query_embedding, k=1)

    best_score = D[0][0]
    best_answer = knowledge_base[I[0][0]] if D[0][0] < 1.5 else "No relevant answer found."

    if input_lang != 'en':
        best_answer = translate_from_en(best_answer, input_lang)

    return best_answer, f"{round(1 / (1 + best_score), 2)} confidence"

# Initialize KB on load
try:
    knowledge_texts = load_knowledge()
    if knowledge_texts:
        kb_index, knowledge_base = build_vector_store(knowledge_texts)
    else:
        knowledge_base = []
except Exception as e:
    knowledge_base = []
