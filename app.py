import streamlit as st
from llm_backend import (
    load_model,
    load_translator,
    load_available_languages,
    detect_language,
    translate_text,
    load_knowledge_base,
    answer_question,
)

st.set_page_config(page_title="Multilingual LLM Q&A", layout="centered")

st.title("🌍 Multilingual Q&A Assistant")
st.markdown("Ask any question in your native language. The system will understand and reply!")

model = load_model()
translator = load_translator()
supported_languages = load_available_languages()
texts, index, embeddings = load_knowledge_base(model)

user_input = st.text_input("🔍 Enter your question:")

if st.button("Get Answer"):
    if not user_input.strip():
        st.warning("Please enter a question.")
    else:
        src_lang = detect_language(user_input)
        st.write(f"🗣️ Detected language: `{src_lang}`")

        translated_input = translate_text(user_input, src_lang, "en")
        answer = answer_question(translated_input, model, texts, index, embeddings)

        translated_answer = translate_text(answer, "en", src_lang)
        st.success(f"✅ Answer: {translated_answer}")
