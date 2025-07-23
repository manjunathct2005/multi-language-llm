import streamlit as st
from llm_backend import process_input, knowledge_base
import warnings
import re

# Suppress warnings
warnings.filterwarnings("ignore")

# Streamlit page setup
st.set_page_config(page_title="📚 Multilingual Knowledge Base", layout="wide")
st.title("💡 Multilingual Knowledge Base Assistant")
st.markdown("Ask in **Telugu** or **English**. You'll get clean responses from your `.txt` knowledge base.")

# Knowledge base load check
if not knowledge_base:
    st.error("❌ Knowledge base not loaded.")
else:
    st.success(f"✅ {len(knowledge_base)} knowledge blocks loaded.")

# Input field
query = st.text_area("Ask your question here:", height=100)
answer_type = st.radio("Choose Response Style:", ["Summary", "Detailed (Chat-style)"], horizontal=True)

# Submit button
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
            else:
                st.markdown("💬 **Detailed Answer:**")

            st.markdown(answer)
