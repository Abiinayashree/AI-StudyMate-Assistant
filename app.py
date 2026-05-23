import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.title("AI StudyMate Assistant")

st.subheader("Semantic Search & Retrieval System")

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

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = model.encode(chunks)

embedding_array = np.array(embeddings)

# Create FAISS index
dimension = embedding_array.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embedding_array)

# User input
query = st.text_input("Ask Question From PDF")

if query:

    # Query embedding
    query_embedding = model.encode([query])

    # Search
    k = 2

    distances, indices = index.search(
        np.array(query_embedding),
        k
    )

    st.subheader("Top Matching Results")

    for i in indices[0]:
        st.write(chunks[i])
        st.write("-------------------")