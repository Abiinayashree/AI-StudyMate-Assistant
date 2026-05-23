from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

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

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = model.encode(chunks)

# Convert to numpy array
embedding_array = np.array(embeddings)

# Create FAISS index
dimension = embedding_array.shape[1]

index = faiss.IndexFlatL2(dimension)

# Store embeddings
index.add(embedding_array)

# User query
query = "What is Generative AI?"

# Convert query into embedding
query_embedding = model.encode([query])

# Search similar chunks
k = 2

distances, indices = index.search(
    np.array(query_embedding),
    k
)

# Display results
print("User Query:", query)

print("\nTop Matching Chunks:\n")

for i in indices[0]:
    print(chunks[i])
    print("\n-----------------\n")