import os
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import torch

# ✅ Download NLTK 'punkt' tokenizer if not already available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ✅ Load transcripts from folder
def load_transcripts(transcript_folder="transcripts"):
    texts = []
    if not os.path.exists(transcript_folder):
        print(f"[INFO] Transcript folder not found: {transcript_folder}")
        return []

    for filename in os.listdir(transcript_folder):
        if filename.endswith(".txt"):
            with open(os.path.join(transcript_folder, filename), "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    texts.append(content)
    return texts

# ✅ Load transcripts once at startup so app.py can import
texts = load_transcripts()

# ✅ Answer question by finding the best matching content
def answer_question(question, texts):
    if not texts:
        return "⚠️ No transcript data found."

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Combine all texts and tokenize
    full_text = " ".join(texts).replace("\n", " ")
    sentences = sent_tokenize(full_text)

    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    question_embedding = model.encode(question, convert_to_tensor=True)

    similarities = util.cos_sim(question_embedding, sentence_embeddings)[0]
    top_k = torch.topk(similarities, k=3)

    # Combine top 3 similar answers
    top_answers = [sentences[idx] for idx in top_k.indices]
    return " ".join(top_answers)
