import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

st.title("AI StudyMate Assistant")

st.subheader("Embeddings Generation System")

# Load PDF
pdf = PdfReader("sample.pdf")

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

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = model.encode(chunks)

# Display embedding count
st.write("Total Embeddings Generated:", len(embeddings))

# Display sample chunk
st.subheader("Sample Chunk")
st.write(chunks[0])

# Display sample embedding
st.subheader("Sample Embedding Vector")
st.write(embeddings[0])