import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.title("AI StudyMate Assistant")

st.subheader("FAISS Vector Database System")

# Load PDF
pdf = PdfReader("documents/sample.pdf")

# Extract text
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
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = model.encode(chunks)

# Convert to numpy
embedding_array = np.array(embeddings)

# Create FAISS index
dimension = embedding_array.shape[1]

index = faiss.IndexFlatL2(dimension)

# Store embeddings
index.add(embedding_array)

# Display results
st.write("Total Chunks:", len(chunks))

st.write("Total Embeddings Stored:", index.ntotal)

# Display sample chunk
st.subheader("Sample Chunk")

st.write(chunks[0])

# Display sample embedding
st.subheader("Sample Embedding Vector")

st.write(embeddings[0][:10])