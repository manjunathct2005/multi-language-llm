import streamlit as st
from llm_backend import load_transcripts, answer_question

st.set_page_config(page_title="Multilingual Q&A App", layout="centered")
st.markdown("## 🧠 Multilingual Q&A App")
st.markdown("Ask your question in any language (English, Hindi, Telugu, etc.)")

# Load transcripts
with st.spinner("Loading transcript data..."):
    try:
        sentences, embeddings = load_transcripts()
        st.success("Transcript data loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

# User input
query = st.text_input("Enter your question below:")

if st.button("Get Answer"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Processing..."):
                response = answer_question(query, sentences, embeddings)
            st.success("✅ Answer:")
            st.write(response)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
