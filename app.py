import streamlit as st
from llm_backend import answer_question, texts

# Page setup
st.set_page_config(page_title="🧠 Multilingual Q&A", layout="centered")
st.markdown("<h2 style='text-align:center;'>💬 Chat-Style Q&A (No Models)</h2>", unsafe_allow_html=True)

# Session state for conversation history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box for user query
user_input = st.text_area("Ask a question (in any language):", height=100, placeholder="Type your question here...")

# Handle submit
if st.button("🔍 Get Answer"):
    if user_input.strip():
        # Get the answer
        answer = answer_question(user_input.strip(), texts)

        # Save in session history
        st.session_state.chat_history.append(("You", user_input.strip()))
        st.session_state.chat_history.append(("Bot", answer))
    else:
        st.warning("❗ Please enter a question before submitting.")

# Display chat history
for role, msg in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"**🧑‍💬 You:** {msg}")
    else:
        st.markdown(f"**🤖 Bot:** {msg}")
