# D:/hindupur_dataset/generate_embeddings.py

import os
import torch
from sentence_transformers import SentenceTransformer

TRANSCRIPT_DIR = "D:/hindupur_dataset/transcripts"
EMBEDDINGS_PATH = "D:/hindupur_dataset/embeddings.pt"

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = []

print("📁 Scanning transcript files...")

for file in os.listdir(TRANSCRIPT_DIR):
    if file.endswith(".txt"):
        path = os.path.join(TRANSCRIPT_DIR, file)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                texts.append(content)
                print(f"✅ Loaded: {file}")

if not texts:
    print(f"❌ No transcript texts found in: {TRANSCRIPT_DIR}")
    exit()

embeddings = model.encode(texts, convert_to_tensor=True)
torch.save({"embeddings": embeddings, "texts": texts}, EMBEDDINGS_PATH)
print(f"\n✅ Saved {len(texts)} embeddings to: {EMBEDDINGS_PATH}")
