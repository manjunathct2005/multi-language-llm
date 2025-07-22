import os
import streamlit as st
from llm_backend import process_input, knowledge_base, detect_language

st.set_page_config(page_title="Multilingual Knowledge Base Assistant", layout="centered")
st.title("💡 Multilingual Knowledge Base Assistant")
st.markdown("Ask in **Telugu**, **Hindi**, **Kannada**, or **English**. Clean answers from `.txt` transcripts.")

# Load knowledge base
with st.spinner("Loading knowledge base..."):
    texts, index, embeddings = knowledge_base()
st.success(f"Knowledge base loaded ✅ ({len(texts)} blocks)")

# Input section
st.markdown("### 🔎 Ask your question here:")
user_query = st.text_input("", placeholder="Ask your question...")

# Response style toggle
st.markdown("#### 📝 Response Style:")
style = st.radio("", ["Summary", "Detailed (Chat-style)"], index=0)

# Submit button
if st.button("🚀 Get Answer"):
    if user_query.strip() == "":
        st.warning("Please enter a question.")
    else:
        lang = detect_language(user_query)

        if lang not in ["en", "hi", "te", "kn"]:
            st.error("❌ Only Telugu/English/Hindi/Kannada questions are supported.")
        else:
            with st.spinner("Processing your question..."):
                answer = process_input(user_query, style.lower(), texts, index, embeddings, lang)
            st.markdown("### ✅ Answer:")
            st.write(answer)

# Footer
st.markdown("---")
st.markdown("🔍 *Only `.txt` files from the knowledge base folder are used. Offline, multilingual QA.*")
