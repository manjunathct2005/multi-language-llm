import os
import streamlit as st
from llm_backend import process_input

# Set up Streamlit UI
st.set_page_config(page_title="Multilingual LLM", layout="centered")
st.markdown("<h2 style='text-align: center;'>🌐 Multilingual LLM Tool</h2>", unsafe_allow_html=True)

# Input box
query = st.text_input("💬 Enter your question:", placeholder="Type here in Hindi, Telugu, Kannada, or English...")

# Button
if st.button("Ask"):
    if not query.strip():
        st.warning("⚠️ Please enter a valid question.")
    else:
        try:
            # Get response from backend
            answer, source = process_input(query)

            if answer.startswith("❌") or answer.startswith("Sorry") or "not found" in answer:
                st.markdown(f"<p style='color:red;font-weight:bold;'>🔍 {answer}</p>", unsafe_allow_html=True)
            else:
                st.success("✅ Answer:")
                st.write(answer)
                if source:
                    st.markdown(f"<small>📁 Source: <code>{source}</code></small>", unsafe_allow_html=True)
        except Exception as e:
            st.error("❌ Error occurred while processing your question.")
            st.text(str(e))

# Optional UI footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;'>Built by Manju Nath 🚀</div>", unsafe_allow_html=True)
