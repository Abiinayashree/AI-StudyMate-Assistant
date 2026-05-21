from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

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

# Print total embeddings
print("Total Embeddings:", len(embeddings))

# Print first embedding vector
print("\nFirst Embedding Vector:\n")

print(embeddings[0])