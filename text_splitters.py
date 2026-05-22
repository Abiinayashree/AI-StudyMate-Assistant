from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
pdf = PdfReader("documents/sample.pdf")

# Extract text
text = ""

for page in pdf.pages:
    text += page.extract_text()

# Initialize text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Generate chunks
chunks = splitter.split_text(text)

# Print total chunks
print("Total Chunks:", len(chunks))

# Print sample chunk
print("\nSample Chunk:\n")

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---\n")
    print(chunk)