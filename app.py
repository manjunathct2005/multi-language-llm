import streamlit as st
from llm_backend import detect_language, translate_text, load_knowledge_base, get_answer

st.set_page_config(page_title="Multilingual LLM Tool", layout="centered")
st.title("🌐 Multilingual Q&A Assistant")
st.markdown("Supports questions in English, Hindi, Telugu, and Kannada.")

texts, embeddings = load_knowledge_base()

query = st.text_input("🔍 Ask your question")

if st.button("Get Answer") and query:
    lang = detect_language(query)
    st.markdown(f"🗣️ Detected Language: **{lang.upper()}**")

    translated_query = translate_text(query, lang, "en") if lang != "en" else query
    answer = get_answer(translated_query, texts, embeddings)

    if answer:
        final_answer = translate_text(answer, "en", lang) if lang != "en" else answer
        st.success(final_answer)
    else:
        st.warning("🤷‍♂️ No relevant answer found.")
