# === Auto-install required packages if missing ===
import os
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = [
    "faiss-cpu==1.7.4",
    "sentence-transformers",
    "transformers",
    "torch",
    "torchaudio",
    "langdetect",
    "gTTS",
    "pydub",
    "translate",
    "streamlit"
]

for package in required_packages:
    try:
        __import__(package.split("==")[0])
    except ImportError:
        install(package)

# === Your Streamlit App Starts Here ===
import streamlit as st
from llm_backend import process_input, knowledge_base
import warnings
import re

warnings.filterwarnings("ignore")

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
