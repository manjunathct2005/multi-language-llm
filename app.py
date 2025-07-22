import os
import streamlit as st
from llm_backend import process_input

# === Streamlit UI Configuration ===
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

st.set_page_config(page_title="📚 Multilingual LLM Tool", layout="centered")
st.title("🔍 Multilingual Q&A using Your Text Files")

# === Sidebar ===
st.sidebar.title("📁 Settings")
st.sidebar.markdown("This tool answers your questions using knowledge from uploaded `.txt` files.")

# === Input ===
user_input = st.text_input("📝 Ask a question (in English, Hindi, Telugu, Kannada):", "What is the data science lifecycle?")

# === Process Input ===
if user_input.strip():
    with st.spinner("Thinking..."):
        response = process_input(user_input)

    # === Output in Styled Format ===
    if response:
        st.markdown("""
        <div style="padding:1em; background-color:#f1f1f1; border-radius:10px">
        <h4 style="color:#4CAF50;">📘 Answer</h4>
        <div style="font-size:16px; line-height:1.6">{}</div>
        </div>
        """.format(response), unsafe_allow_html=True)
    else:
        st.warning("❌ Sorry, I couldn't find an answer. Please try rephrasing your question.")
else:
    st.info("💡 Enter a question above to get started.")

# === Footer ===
st.markdown("---")
st.markdown("Built with ❤️ using HuggingFace models and Sentence Transformers")
