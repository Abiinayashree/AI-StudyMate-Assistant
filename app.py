import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import faiss
import numpy as np
import os

# Load environment variables
load_dotenv()

# Streamlit title
st.title("AI StudyMate Assistant")

st.subheader("AI-Powered PDF Question Answering System")

# Read PDF
pdf = PdfReader("documents/sample.pdf")

text = ""

for page in pdf.pages:
    text += page.extract_text()

# Split text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

# Embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = embedding_model.encode(chunks)

embedding_array = np.array(embeddings)

# FAISS database
index = faiss.IndexFlatL2(
    embedding_array.shape[1]
)

index.add(embedding_array)

# Load Groq model
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile"
)

# User input
query = st.text_input("Ask Question From PDF")

if query:

    # Query embedding
    query_embedding = embedding_model.encode([query])

    # Retrieve chunks
    distances, indices = index.search(
        np.array(query_embedding),
        2
    )

    retrieved_text = ""

    for i in indices[0]:
        retrieved_text += chunks[i]

    # Prompt
    prompt = f"""
    Answer the question based on the context below.

    Context:
    {retrieved_text}

    Question:
    {query}
    """

    # AI response
    response = llm.invoke(prompt)

    # Display answer
    st.subheader("AI Answer")

    st.write(response.content)