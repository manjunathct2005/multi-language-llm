# app.py

import streamlit as st
from llm_backend import process_input

# === PAGE CONFIG ===
st.set_page_config(page_title="📚 Smart Multilingual Q&A", layout="centered")
st.title("🤖 Multilingual Knowledge Assistant")
st.write("Ask questions from your custom data in **English, Hindi, Telugu, or Kannada**.")

# === INPUT SECTION ===
user_input = st.text_input("💬 Enter your question below:", max_chars=500, placeholder="e.g., What are the steps in Data Science lifecycle?")

# === PROCESS AND DISPLAY ===
if user_input:
    with st.spinner("🔍 Searching your data for the most relevant answer..."):
        response = process_input(user_input)

    # === FORMATTED RESPONSE DISPLAY ===
    st.markdown("---")
    st.subheader("📝 Answer:")
    
    # Split the answer into sections (headings, bullets, code, etc.)
    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith("###"):
            st.markdown(f"{line}")
        elif line.startswith("**") and line.endswith("**"):
            st.markdown(f"{line}")
        elif line.startswith("- ") or line.startswith("• "):
            st.markdown(f"{line}")
        elif line.startswith("```") and line.endswith("```"):
            st.code(line.strip("```"), language="python")
        elif "```" in line:
            # Handle multiline code blocks
            code_block = []
            code_mode = False
            for l in lines:
                if l.strip().startswith("```"):
                    if not code_mode:
                        code_mode = True
                        code_lang = l.strip()[3:] or "text"
                    else:
                        st.code("\n".join(code_block), language=code_lang)
                        code_block = []
                        code_mode = False
                elif code_mode:
                    code_block.append(l)
            break
        else:
            st.markdown(f"{line}")
