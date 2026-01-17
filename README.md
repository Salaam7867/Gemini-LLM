📄 Enterprise Document RAG (Gemini)
Overview

This project is a production-style Retrieval-Augmented Generation (RAG) system that allows users to query PDF documents using a large language model (Gemini) with grounded, source-based answers.

The system is designed to reflect real-world GenAI architecture, separating document retrieval from LLM reasoning.

Key Features

PDF ingestion and preprocessing

Chunking with overlap for better recall

Semantic search using vector embeddings (FAISS)

Context-grounded responses using Gemini LLM

Source citation for transparency

Deployed on Streamlit Cloud

System Architecture
PDF → Text Chunks → Embeddings → Vector Store
                     ↓
                 Retriever
                     ↓
               Context + Question
                     ↓
                 Gemini LLM
                     ↓
                   Answer

Tech Stack

LLM: Gemini (Google Generative AI)

Embeddings: sentence-transformers (MiniLM)

Vector DB: FAISS

Framework: LangChain

Frontend: Streamlit

Language: Python

Why Gemini?

Local instruction models (e.g., FLAN-T5) are inconsistent for reasoning-heavy tasks.
Gemini provides:

Stable reasoning

Better instruction following

Production-grade reliability

This reflects how most real-world GenAI systems are built.

Limitations

No authentication or rate limiting (demo scope)

Evaluation metrics not yet automated

Designed for single-user demo usage

These are intentional tradeoffs for internship-scale projects.

How to Run Locally
pip install -r requirements.txt
streamlit run app.py

Use Cases

Resume screening

Policy document Q&A

Internal knowledge assistants

Academic document analysis

What This Project Demonstrates

Understanding of RAG architecture

Practical LLM integration

Awareness of deployment constraints

System-level GenAI thinking (not just demos)
