import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import faiss
import numpy as np

# Load environment variables
load_dotenv()

# Streamlit UI
st.title("AI StudyMate Assistant")

st.caption("AI-powered Conversational PDF Assistant using RAG")

st.subheader("Conversational AI PDF Chatbot")

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load PDF
pdf = PdfReader("documents/sample.pdf")

# Extract text
text = ""

for page in pdf.pages:
    text += page.extract_text()

# Text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Create chunks
chunks = splitter.split_text(text)

# Embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Generate embeddings
embeddings = embedding_model.encode(chunks)

embedding_array = np.array(embeddings)

# Create FAISS index
index = faiss.IndexFlatL2(
    embedding_array.shape[1]
)

# Store embeddings
index.add(embedding_array)

# Load Groq LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile"
)

# User input
query = st.chat_input("Ask Question From PDF")

if query:

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    # Convert query to embedding
    query_embedding = embedding_model.encode([query])

    # Retrieve relevant chunks
    distances, indices = index.search(
        np.array(query_embedding),
        2
    )

    # Store retrieved text
    retrieved_text = ""

    for i in indices[0]:
        retrieved_text += chunks[i] + "\n\n"

    # Conversation history
    history = ""

    for msg in st.session_state.messages:
        history += f"{msg['role']}: {msg['content']}\n"

    # Prompt
    prompt = f"""
    You are an AI Study Assistant.

    Previous Conversation:
    {history}

    Context:
    {retrieved_text}

    User Question:
    {query}

    Give a helpful answer based on the context.
    """

    # Generate AI response
    response = llm.invoke(prompt)

    answer = response.content

    # Store AI answer
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # Store retrieved context
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": f"Retrieved Context:\n\n{retrieved_text}"
        }
    )

# Display conversation
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])