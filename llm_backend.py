import streamlit as st
from llm_backend import load_texts_and_embeddings, detect_language, translate_to_english, translate_from_english, get_answer

st.set_page_config(page_title="Multilingual Q&A", layout="centered")
st.title("🧠 Multilingual Question Answering")

@st.cache_resource
def load_resources():
    return load_texts_and_embeddings()

texts, index = load_resources()

user_input = st.text_input("Enter your question (any language):")

if user_input:
    lang = detect_language(user_input)
    translated = translate_to_english(user_input)
    answer_en = get_answer(translated, texts, index)
    final_answer = translate_from_english(answer_en, lang)

    st.markdown(f"**Answer ({lang.upper()}):** {final_answer}")
