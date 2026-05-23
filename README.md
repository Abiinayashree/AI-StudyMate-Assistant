# AI StudyMate Assistant

AI StudyMate Assistant is an AI-powered educational document intelligence system built using LangChain, Groq API, Streamlit, and FAISS.

The application allows users to upload study materials or PDF documents and generate AI-powered summaries, important notes, quiz questions, and contextual answers using Retrieval-Augmented Generation (RAG) architecture.

## Technologies Used

- Python
- LangChain
- Streamlit
- Groq API
- FAISS
- PyPDF
- HuggingFace Embeddings

## Modules Completed

### PDF Text Extraction System

Implemented PDF document loading and text extraction using PyPDF. The system reads educational PDF documents and displays extracted content through the Streamlit interface.

### Text Splitting & Chunk Generation System

Implemented LangChain RecursiveCharacterTextSplitter for splitting large PDF text into smaller chunks. This workflow prepares document data for embeddings and semantic retrieval processes.

### FAISS Vector Database Integration

Implemented vector database integration using FAISS for storing semantic embeddings generated from document chunks. This module forms the retrieval foundation for Retrieval-Augmented Generation (RAG) workflows.

### Semantic Search & Retrieval System

Implemented semantic retrieval workflow using FAISS vector database and Sentence Transformers for context-aware document search and intelligent chunk retrieval.