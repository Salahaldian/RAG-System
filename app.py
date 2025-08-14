import os
import streamlit as st
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# -----------------------------
# إعدادات
# -----------------------------

load_dotenv()  # تحميل متغيرات البيئة
HF_API_KEY = os.getenv("HF_API_KEY")  # ضيف API Key تبع Hugging Face في ملف .env

# موديل Mistral
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# Hugging Face Inference Client
client = InferenceClient(
    MODEL_ID,
    token=HF_API_KEY
)

# Sentence Transformer للـ Embeddings
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# دوال مساعدة
# -----------------------------

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def create_faiss_index(chunks):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

def retrieve_relevant_chunks(query, chunks, index, top_k=3):
    query_emb = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, top_k)
    return [chunks[i] for i in indices[0]]

def query_huggingface_api(prompt):
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        response = client.chat_completion(messages, max_tokens=500)
        return response.choices[0].message["content"]
    except Exception as e:
        return f"❌ API Error: {str(e)}"

def rag_pipeline(question, chunks, index):
    retrieved_chunks = retrieve_relevant_chunks(question, chunks, index)
    context = "\n".join(retrieved_chunks)
    prompt = f"""
Use the provided context to answer the question.

Context:
{context}

Question: {question}
Answer:
"""
    return query_huggingface_api(prompt)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="📚 Document QA System", layout="wide")
st.title("📚 Document QA with Mistral + FAISS")

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(text)
    index, _ = create_faiss_index(chunks)
    st.success("✅ Document processed and indexed!")

    question = st.text_input("Ask a question about your document:")
    if st.button("Get Answer") and question:
        with st.spinner("Thinking..."):
            answer = rag_pipeline(question, chunks, index)
        st.markdown("### Answer:")
        st.write(answer)

        # عرض الـ chunks المسترجعة (للتوضيح)
        # st.markdown("### 🔍 Retrieved Chunks:")
        # for i, chunk in enumerate(retrieve_relevant_chunks(question, chunks, index), 1):
        #     st.markdown(f"**Chunk {i}:** {chunk}")