import streamlit as st
from llm_backend import process_input

# === Page Configuration ===
st.set_page_config(page_title="📚 Smart Multilingual Q&A", layout="centered")

# === Title and Instructions ===
st.title("📚 Smart Multilingual Q&A")
st.markdown("### 🤖 Multilingual Knowledge Assistant")
st.markdown("""
Ask questions from your notes in:
- **English**
- **Hindi**
- **Telugu**
- **Kannada**

You will get precise, clean answers — no extra info!
""")

# === Input Field ===
user_input = st.text_input("💬 Enter your question:", max_chars=500)

# === Process and Display Response ===
if user_input:
    with st.spinner("🔍 Searching your knowledge base..."):
        response = process_input(user_input).strip()

    st.markdown("---")
    st.markdown("### 🧠 Answer")
    
    # Highlight answer in red using HTML styling
    st.markdown(
        f"<div style='background-color:#fff0f0;padding:10px;border-left:4px solid red;color:#b00020;font-family:monospace'>{response}</div>",
        unsafe_allow_html=True
    )
