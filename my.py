import os
import torch
from sentence_transformers import SentenceTransformer

# === CONFIGURATION ===
TEXT_FOLDER = r"C:\Users\manjunath\OneDrive\Desktop\my1"
EMBEDDING_FILE = r"D:\hindupur_dataset\embeddings1.pt"
MODEL_NAME = "all-MiniLM-L6-v2"

# === Load Pretrained SentenceTransformer Model ===
model = SentenceTransformer(MODEL_NAME)

# === Load Existing Embeddings ===
if os.path.exists(EMBEDDING_FILE):
    data = torch.load(EMBEDDING_FILE)
    existing_texts = data.get("texts", [])
    existing_embeddings = data.get("embeddings", torch.empty(0))
else:
    existing_texts = []
    existing_embeddings = torch.empty((0, model.get_sentence_embedding_dimension()))

# === Read & Clean New Text Files ===
new_texts = []
for file_name in os.listdir(TEXT_FOLDER):
    if file_name.endswith(".txt"):
        file_path = os.path.join(TEXT_FOLDER, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content and content not in existing_texts:
                new_texts.append(content)

# === Generate New Embeddings ===
if new_texts:
    new_embeddings = model.encode(new_texts, convert_to_tensor=True)
    updated_texts = existing_texts + new_texts
    updated_embeddings = torch.cat([existing_embeddings, new_embeddings], dim=0)

    # === Save Back to File ===
    torch.save({
        "texts": updated_texts,
        "embeddings": updated_embeddings
    }, EMBEDDING_FILE)

    print(f"[✓] Added {len(new_texts)} new texts. Total KB size: {len(updated_texts)}")
else:
    print("[!] No new unique .txt files found or all are already added.")
