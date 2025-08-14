# 📚 Document QA System (RAG with Mistral + FAISS)

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that allows users to upload a **PDF document** and ask questions about its content. The system retrieves the most relevant text chunks using **FAISS** and then generates an answer using **Mistral-7B-Instruct** hosted on Hugging Face.

---

## 🚀 Features
- 📂 Upload any PDF document.
- 🔍 Automatically splits text into **chunks** for retrieval.
- ⚡ Uses **FAISS** for efficient similarity search.
- 🤖 Answers questions with **Mistral-7B-Instruct** (via Hugging Face API).
- 📝 Displays both the **final answer** and the **retrieved chunks**.

---

## 🛠️ Tech Stack
- [Streamlit](https://streamlit.io/) → Web interface
- [FAISS](https://github.com/facebookresearch/faiss) → Vector search
- [Sentence Transformers](https://www.sbert.net/) → Text embeddings
- [Mistral-7B-Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) → LLM for text generation
- [Hugging Face Hub](https://huggingface.co/) → Model inference API

---

## 📂 How it Works
1. Upload a PDF file.
2. The text is extracted and split into **chunks**.
3. FAISS indexes the chunks for similarity search.
4. When you ask a question:
   - The system retrieves the **most relevant chunks**.
   - A prompt is sent to **Mistral** including these chunks as context.
   - The model generates an **answer**.
5. The retrieved chunks are displayed for transparency.

---
