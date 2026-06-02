### AI StudyMate Assistant

# Overview

AI StudyMate Assistant is an AI-powered educational document intelligence system built using Streamlit, LangChain, Groq API, FAISS, and Retrieval-Augmented Generation (RAG) architecture.

The application enables users to upload educational PDF documents and interact with them through AI-powered question answering, document summarization, important notes extraction, and quiz generation.

---

# Project Objective

Students often spend significant time reading lengthy study materials, preparing notes, and searching for important questions. This project simplifies the learning process using AI-powered document understanding and retrieval techniques.

---

# Technologies Used

- Python
- Streamlit
- LangChain
- Groq API
- FAISS Vector Database
- Sentence Transformers
- PyPDF
- NumPy
- Python Dotenv

---

# System Architecture

User → PDF Upload → Text Extraction → Text Splitting → Embedding Generation → FAISS Vector Storage → Semantic Retrieval → Groq LLM → Streamlit Interface

---

# Features

- Multi PDF Upload
- PDF Text Extraction
- Text Chunking
- Embedding Generation
- FAISS Vector Database
- Semantic Search
- AI Question Answering
- AI Summary Generation
- Important Notes Extraction
- Quiz Generation
- Conversational Memory
- Chat Statistics Dashboard
- Streamlit User Interface

---

# AI Concepts Used

- Generative AI
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)
- Embeddings
- Vector Databases
- Semantic Search
- Prompt Engineering

---

# Project Workflow

1. User uploads PDF documents.
2. PyPDF extracts text from documents.
3. LangChain splits text into chunks.
4. Sentence Transformers generate embeddings.
5. FAISS stores embeddings as vectors.
6. User submits a query.
7. Semantic search retrieves relevant chunks.
8. Groq LLM generates context-aware responses.
9. Streamlit displays the output.

---

# Requirements

Before running the project, ensure the following requirements are met.

# Software Requirements

- Python 3.10 or above
- Visual Studio Code (Recommended)
- Git
- GitHub Account
- Groq API Key

---

# Installation

1. Clone Repository

git clone https://github.com/Abiinayashree/AI-StudyMate-Assistant.git

2. Install Dependencies

pip install -r requirements.txt

3. Create .env File

GROQ_API_KEY=your_api_key

4. Run Application

streamlit run app.py

---

# Project Structure

AI-StudyMate-Assistant/

├── documents/
│   └── sample.pdf

├── .env
├── app.py
├── embeddings.py
├── pdf_extraction.py
├── text_splitters.py
├── vector_store.py
├── retrieval.py
├── requirements.txt
└── README.md

---

# Project Status

✅ Successfully Developed and Tested

# Implemented Modules:

- PDF Processing and Text Extraction
- Text Chunking and Splitting
- Embedding Generation
- FAISS Vector Database Integration
- Semantic Search and Retrieval
- Retrieval-Augmented Generation (RAG) Workflow
- AI-Powered Question Answering
- AI Summary Generation
- Important Notes Extraction
- Quiz Generation
- Conversational Memory
- Interactive Streamlit User Interface

---

# Key Highlights

✔ AI-Powered Educational Document Intelligence System

✔ Retrieval-Augmented Generation (RAG) Based Architecture

✔ Semantic Search Using Vector Embeddings

✔ Multi-PDF Processing Capability

✔ Context-Aware AI Question Answering

✔ Automated Summary and Notes Generation

✔ Dynamic Quiz Generation

✔ Interactive Streamlit Web Interface

---

# Future Enhancements

- Download Summary as PDF
- Download Notes as PDF
- Advanced Quiz Generation
- Voice-Based Interaction
- Multi-Language Support
- Cloud Deployment
- User Authentication
- Learning Progress Tracking

---

# Author

Abinayashree M

Postgraduate Student

Developer of AI StudyMate Assistant

Passionate about Artificial Intelligence, Generative AI, Large Language Models (LLMs), and Educational Technology Solutions.