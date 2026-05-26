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

# Streamlit title
st.title("AI StudyMate Assistant")

st.subheader("Conversational AI PDF Assistant")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load PDF
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

# FAISS index
index = faiss.IndexFlatL2(
    embedding_array.shape[1]
)

index.add(embedding_array)

# Groq model
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile"
)

# User input
query = st.chat_input("Ask Question From PDF")

if query:

    # Store user message
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

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

    Give a helpful answer.
    """

    # AI response
    response = llm.invoke(prompt)

    answer = response.content

    # Store AI response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

# Display chat
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])