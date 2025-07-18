import streamlit as st
from llm_backend import process_input, knowledge_base
import warnings
import re
import os
import gdown

warnings.filterwarnings("ignore")

# ========== CONFIG ==========
MODEL_FOLDER_PATH = "models"
GOOGLE_DRIVE_FOLDER_ID = "https://drive.google.com/drive/folders/19q5O6vdFvOpAAFlZ7epIplGEGIPMYzZl?usp=drive_link"  # <- Replace this!
MODEL_FILES = [
    "whisper-base/config.json",
    "whisper-base/pytorch_model.bin",
    "sentence-transformers/all-MiniLM-L6-v2/config.json",
    "sentence-transformers/all-MiniLM-L6-v2/pytorch_model.bin",
    # Add more files if needed...
]

# ========== DOWNLOAD MODELS FROM GOOGLE DRIVE ==========

@st.cache_resource
def download_models():
    if not os.path.exists(MODEL_FOLDER_PATH):
        os.makedirs(MODEL_FOLDER_PATH)

    for file_path in MODEL_FILES:
        local_path = os.path.join(MODEL_FOLDER_PATH, file_path)
        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            file_name = os.path.basename(file_path)
            gdown.download(
                f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FOLDER_ID}&export=download&confirm=t",
                output=local_path,
                quiet=False,
                fuzzy=True
            )
    return True

download_models()

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="📚 Multilingual Knowledge Base", layout="wide")
st.title("💡 Multilingual Knowledge Base Assistant")
st.markdown("Ask in **Telugu** or **English**. You'll get clean responses from your `.txt` knowledge base.")

if not knowledge_base:
    st.error("❌ Knowledge base not loaded.")
else:
    st.success(f"✅ {len(knowledge_base)} knowledge blocks loaded.")

query = st.text_area("Ask your question here:", height=100)
answer_type = st.radio("Choose Response Style:", ["Summary", "Detailed (Chat-style)"], horizontal=True)

if st.button("🔍 Get Answer"):
    if not query.strip():
        st.warning("⚠️ Please enter a question.")
    elif not knowledge_base:
        st.error("❌ Knowledge base is empty.")
    else:
        with st.spinner("Searching your knowledge base..."):
            answer, info = process_input(query)

        if "Only Telugu" in answer or "empty" in answer:
            st.error(f"❌ {answer}")
        elif "No relevant answer" in answer:
            st.warning(f"⚠️ {answer}")
        else:
            st.markdown(f"### ✅ Answer (Confidence: {info})")
            if answer_type == "Summary":
                st.markdown("📘 **Summary:**")
                for line in answer.split("\n"):
                    if line.strip().startswith("-") or re.match(r"^\d+\.", line.strip()):
                        st.markdown(f"- {line.strip()}")
                    else:
                        st.write(line.strip())
            else:
                st.markdown("🧾 **Detailed Response:**")
                for para in answer.split("\n\n"):
                    para = para.strip()
                    if para.startswith("```") and para.endswith("```"):
                        st.code(para.strip("```"))
                    else:
                        st.markdown(para)
