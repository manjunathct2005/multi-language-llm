import streamlit as st
from llm_backend import process_input, knowledge_base
import warnings

warnings.filterwarnings("ignore")
st.set_page_config(page_title="📚 Multilingual Knowledge Base", layout="wide")
st.title("💡 Multilingual Knowledge Base Assistant")
st.markdown("Ask in **Telugu**, **Hindi**, or **English**.")

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
            answer, similarity = process_input(query)

        if "No relevant answer" in answer:
            st.warning(f"⚠️ {answer}")
        else:
            st.markdown("### ✅ Answer:")
            st.markdown(f"📌 **Similarity Score**: `{similarity}`")
            st.markdown(f"💬 **Response**: \n\n{answer}")
