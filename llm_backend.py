import os
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
import torch

# ✅ Auto-download NLTK punkt tokenizer if missing
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


# ✅ Load and clean all transcripts from 'transcripts' folder
def load_transcripts(transcript_folder="my1"):
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


# ✅ Answer a question using semantic similarity and paragraph grouping
def answer_question(question, texts):
    if not texts:
        return "⚠️ No transcript data found."

    # Load transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Combine and tokenize all texts
    full_text = " ".join(texts).replace("\n", " ")
    sentences = sent_tokenize(full_text)

    # Create sentence embeddings
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    question_embedding = model.encode(question, convert_to_tensor=True)

    # Compute similarity scores
    similarities = util.cos_sim(question_embedding, sentence_embeddings)[0]
    top_k = torch.topk(similarities, k=3)

    # Get top 3 similar sentences and combine them as a paragraph
    top_answers = [sentences[idx] for idx in top_k.indices]
    return " ".join(top_answers)
