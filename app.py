import re
import warnings
import streamlit as st
from llm_backend import process_input, knowledge_base

warnings.filterwarnings("ignore")
st.set_page_config(page_title="📚 Multilingual Knowledge Base", layout="wide")
st.title("💡 Multilingual Knowledge Base Assistant")
st.markdown("Ask in **Telugu** or **English**. It will search `.txt` files in the `my1/` folder.")

if not knowledge_base:
    st.error("❌ No text data found in `my1/` folder.")
else:
    st.success(f"✅ Loaded {len(knowledge_base)} knowledge chunks.")

query = st.text_area("Ask your question:", height=100)
style = st.radio("Response style:", ["Summary", "Chat-style"], horizontal=True)

if st.button("🔍 Search"):
    if not query.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("Searching..."):
            response, confidence = process_input(query)

        if "empty" in response or "Only Telugu" in response:
            st.error(f"❌ {response}")
        elif "No relevant answer" in response:
            st.warning(f"⚠️ {response}")
        else:
            st.markdown(f"### ✅ Answer (Confidence: {confidence})")
            if style == "Summary":
                for line in response.split("\n"):
                    if line.strip():
                        st.markdown(f"- {line.strip()}")
            else:
                st.markdown(response.replace("\n", "\n\n"))
