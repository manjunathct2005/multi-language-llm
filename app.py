import warnings
import os
import re
import streamlit as st
from llm_backend import process_input, knowledge_base  # make sure llm_backend.py is correct and in same folder

# Suppress warnings and TensorFlow logs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Streamlit page config
st.set_page_config(page_title="📚 Multilingual Knowledge Base", layout="wide")
st.title("💡 Multilingual Knowledge Base Assistant")
st.markdown("Ask in **Telugu**, **Hindi**, **Kannada**, or **English**. Clean answers from `.txt` transcripts.")

# Check knowledge base load
if not knowledge_base:
    st.error("❌ Knowledge base not loaded. Please upload valid `.txt` files.")
else:
    st.success(f"✅ Loaded {len(knowledge_base)} knowledge blocks.")

# User input
query = st.text_area("🔎 Ask your question here:", height=100, placeholder="E.g. హిందూపురం ఎవరి పేరు మీద ఉంది?")
answer_type = st.radio("📝 Response Style:", ["Summary", "Detailed (Chat-style)"], horizontal=True)

# Process button
if st.button("🚀 Get Answer"):
    if not query.strip():
        st.warning("⚠️ Please enter a valid question.")
    elif not knowledge_base:
        st.error("❌ Knowledge base is empty. Upload files first.")
    else:
        with st.spinner("🔍 Searching the knowledge base..."):
            answer, confidence = process_input(query)

        # Handling different outcomes
        if "Only Telugu" in answer or "empty" in answer:
            st.error(f"❌ {answer}")
        elif "No relevant answer" in answer:
            st.warning(f"⚠️ {answer}")
        else:
            st.markdown(f"### ✅ Answer (Confidence: `{confidence}`)")
            if answer_type == "Summary":
                st.markdown("📘 **Summary Response:**")
                for line in answer.split("\n"):
                    line = line.strip()
                    if line.startswith("-") or re.match(r"^\d+\.", line):
                        st.markdown(f"- {line}")
                    else:
                        st.write(line)
            else:
                st.markdown("🧾 **Detailed Chat-style Response:**")
                for para in answer.split("\n\n"):
                    para = para.strip()
                    if para.startswith("```") and para.endswith("```"):
                        st.code(para.strip("```"))
                    else:
                        st.markdown(para)
