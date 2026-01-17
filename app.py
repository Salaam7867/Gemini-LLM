import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="PDF RAG with Gemini", layout="wide")
st.title("📄 Chat with PDF – Gemini RAG")

# -----------------------------
# Load embeddings (LOCAL, cheap, fast)
# -----------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# -----------------------------
# Load Gemini LLM
# -----------------------------
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2
    )

llm = load_llm()

# -----------------------------
# Build Vector Store
# -----------------------------
def build_vectorstore(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    loader = PyPDFLoader(path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    os.remove(path)
    return FAISS.from_documents(chunks, embeddings)

# -----------------------------
# Answer generation (CORRECT RAG PROMPT)
# -----------------------------
def generate_answer(context, question):
    prompt = f"""
You are an AI assistant answering questions strictly using the provided document context.

Rules:
- Use ONLY the context below
- Do NOT add external knowledge
- If the answer is not present, say exactly: Not found in document

Context:
{context}

Question:
{question}

Answer:
"""
    response = llm.invoke(prompt)
    return response.content.strip()

# -----------------------------
# UI
# -----------------------------
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Indexing document..."):
        vectorstore = build_vectorstore(uploaded_file)

    question = st.text_input("Ask a question from the document")

    if question:
        docs = vectorstore.similarity_search(question, k=4)
        context = "\n\n".join(d.page_content for d in docs)

        answer = generate_answer(context, question)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")
        for i, d in enumerate(docs, 1):
            st.write(f"Source {i}")
            st.write(d.page_content[:300] + "...")
