# transcribe_chunks_fast.py
import os
import torch
import torchaudio
import numpy as np
import whisper

AUDIO_CHUNK_DIR = r"D:\temp_audio_chunks"
OUTPUT_TXT_DIR = r"D:\transcripts_chunks"
os.makedirs(OUTPUT_TXT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base", device=device)
print(f"🔧 Using device: {device}")

def pad_mel_to_3000(mel):
    # Whisper requires mel shape [80, 3000]
    padded = np.zeros((mel.shape[0], 3000))
    padded[:, :mel.shape[1]] = mel
    return padded

def transcribe_and_save(audio_path):
    base = os.path.basename(audio_path)
    txt_path = os.path.join(OUTPUT_TXT_DIR, base.replace(".mp3", ".txt"))

    if os.path.exists(txt_path):
        print(f"✅ Already exists: {txt_path}")
        return

    try:
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio)
        if mel.shape[1] < 3000:
            mel = pad_mel_to_3000(mel)

        options = whisper.DecodingOptions(language="en", fp16=False)
        result = whisper.decode(model, mel, options)
        text = result.text.strip()

        if text:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"✅ Saved: {txt_path}")
        else:
            print(f"⚠️ Empty transcription: {audio_path}")
    except Exception as e:
        print(f"❌ Error in {audio_path}: {str(e)}")

# Process all chunks
chunk_files = sorted([f for f in os.listdir(AUDIO_CHUNK_DIR) if f.endswith(".mp3")])
print(f"🔍 Found {len(chunk_files)} chunks...")

for fname in chunk_files:
    transcribe_and_save(os.path.join(AUDIO_CHUNK_DIR, fname))

print("🎯 Transcription complete.")
