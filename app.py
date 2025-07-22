# Ensure required modules are installed:
# pip install streamlit sentence-transformers faiss-cpu googletrans==4.0.0-rc1 langdetect

import streamlit as st
from llm_backend import process_input, knowledge_base
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# === Streamlit Config ===
st.set_page_config(page_title="📚 Multilingual Knowledge Base", layout="wide")
st.title("💡 Multilingual Knowledge Base Assistant")
st.markdown("Ask your question in **Telugu** or **English**. This assistant will search your local `.txt` knowledge base and respond in the same language.")

# === Knowledge Base Check ===
if not knowledge_base:
    st.error("❌ Knowledge base not loaded or empty.")
else:
    st.success(f"✅ {len(knowledge_base)} knowledge chunks loaded successfully.")

# === User Input ===
query = st.text_area("📝 Ask a question:", height=100, placeholder="E.g. మనిషి శరీరంలో రక్త ప్రసరణ ఎలా జరుగుతుంది?")
answer_type = st.radio("🎛️ Choose Response Style:", ["Summary", "Detailed (Chat-style)"], horizontal=True)

# === Handle Query ===
if st.button("🔍 Get Answer"):
    if not query.strip():
        st.warning("⚠️ Please enter a question to continue.")
    elif not knowledge_base:
        st.error("❌ No knowledge base found. Please check `D:/hindupur_dataset/my1` folder.")
    else:
        with st.spinner("🔎 Searching your knowledge base..."):
            answer, info = process_input(query)

        if info == "en" or isinstance(info, str):
            score_display = f"🧠 **Match Score:** {info}" if info.replace('.', '', 1).isdigit() else ""
        else:
            score_display = ""

        if answer_type == "Summary":
            st.subheader("✅ Answer (Summary)")
            st.markdown(f"{answer}")
            if score_display:
                st.caption(score_display)
        else:
            st.subheader("🧠 Answer (Chat-style)")
            st.markdown(f"**You asked:** {query}")
            st.markdown(f"**Assistant replied:** {answer}")
            if score_display:
                st.caption(score_display)

# === Footer ===
st.markdown("---")
st.markdown("Built with ❤️ using local `.txt` knowledge files and multilingual embedding search.")
