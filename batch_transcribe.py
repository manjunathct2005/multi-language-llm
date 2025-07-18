# C:/Users/manjunath/OneDrive/Desktop/my/batch_transcribe.py

import os
import torch
import torchaudio
from concurrent.futures import ProcessPoolExecutor
from sentence_transformers import SentenceTransformer
from faster_whisper import WhisperModel

ROOT = r"C:/Users/manjunath/OneDrive/Desktop/my"
AUDIO_DIR = os.path.join(ROOT, "audio")
TEMP_DIR = os.path.join(ROOT, "temp_audio_chunks")
TRANSCRIPT_DIR = os.path.join(ROOT, "transcripts")
EMBEDDINGS_PATH = os.path.join(ROOT, "embeddings.pt")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

kb_texts, kb_embeddings = [], None
if os.path.exists(EMBEDDINGS_PATH):
    try:
        data = torch.load(EMBEDDINGS_PATH)
        kb_texts = data.get("texts", [])
        kb_embeddings = data.get("embeddings", None)
    except:
        pass

def convert_mp3_to_wav(mp3_path, wav_path):
    waveform, sr = torchaudio.load(mp3_path)
    torchaudio.save(wav_path, waveform, sr)

def transcribe_file(file):
    try:
        name = file.rsplit(".", 1)[0]
        mp3_path = os.path.join(AUDIO_DIR, file)
        wav_path = os.path.join(TEMP_DIR, name + ".wav")
        txt_path = os.path.join(TRANSCRIPT_DIR, name + ".txt")

        if os.path.exists(txt_path):
            return "⏩", file, None

        convert_mp3_to_wav(mp3_path, wav_path)

        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(wav_path, beam_size=1, vad_filter=True)
        text = " ".join([seg.text.strip() for seg in segments])

        if not text.strip():
            return "❌", file, None

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        return "✅", file, text
    except Exception as e:
        return "❌", f"{file} ({e})", None

def main():
    files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".mp3")]
    results = {"✅": [], "⏩": [], "❌": []}
    new_texts = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        for status, file, text in executor.map(transcribe_file, files):
            print(status, file)
            results[status].append(file)
            if status == "✅" and text:
                new_texts.append(text)

    if new_texts:
        new_embeds = embed_model.encode(new_texts, convert_to_tensor=True)
        global kb_embeddings, kb_texts
        if kb_embeddings is not None:
            kb_embeddings = torch.cat([kb_embeddings, new_embeds], dim=0)
        else:
            kb_embeddings = new_embeds
        kb_texts.extend(new_texts)
        torch.save({"embeddings": kb_embeddings, "texts": kb_texts}, EMBEDDINGS_PATH)
        print(f"\n💾 Embeddings saved to: {EMBEDDINGS_PATH}")

    for k, v in results.items():
        print(f"{k}: {len(v)} files")

if __name__ == "__main__":
    main()
