# multi-language-llm# 🤖 Multilingual Knowledge Base Assistant

A Streamlit-based multilingual assistant that answers queries from a **locally stored knowledge base** using semantic search and translation pipelines. Designed to preserve and promote **regional knowledge** in Indian languages like **Telugu** and **Hindi**.

---

## 🧠 Project Goals

- Enable people to interact with AI in **regional languages**.
- Preserve **local knowledge** in farming, education, health, and culture.
- Promote **inclusive AI development**.

---

## 🌐 Features

- 🔤 Supports input in **Telugu**, **Hindi**, and **English**
- 🌍 Automatically translates questions to English and answers back in the same language
- 💡 Retrieves relevant answers using **Sentence Transformers** and **FAISS**
- 📁 Uses local `.txt` files from `my1/` as the knowledge source
- 📊 Shows confidence score for each answer
- 🚀 Deployable on **Streamlit Cloud**

---

## 🧰 Tech Stack

- Python 3.10+
- [Streamlit](https://streamlit.io)
- [Sentence-Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- `googletrans` (for translation)
- `langdetect`

---

## 🗃️ Folder Structure

```bash
.
├── app.py                   # Streamlit frontend
├── llm_backend.py          # Backend: embeddings, translation, QA
├── my1/                    # Folder with local knowledge .txt files
├── requirements.txt        # Python dependencies



#Setup Instructions
git clone https://github.com/yourusername/multilingual-llm-assistant.git
cd multilingual-llm-assistant

pip install -r requirements.txt

streamlit run app.py




#project overview video link : https://drive.google.com/file/d/1rbEquq9pJ2lbzNlYpAwgCnBIkK_LRBNM/view?usp=sharing


working model link : https://multi-language-llm-i9bm7k4dfgzh2muypzyxzs.streamlit.app/
