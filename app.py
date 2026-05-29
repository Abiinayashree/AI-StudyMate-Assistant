import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import faiss
import numpy as np
import tempfile

# Load env
load_dotenv()

# Title
st.title("AI StudyMate Assistant")

st.caption("Conversational Multi-PDF RAG Assistant")

# Upload PDF
uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type="pdf",
    accept_multiple_files=True
)

# Chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear chat button
if st.button("Clear Conversation"):

    st.session_state.messages = []

    st.rerun()    

# Process only if files uploaded
if uploaded_files:

    full_text = ""

    # Read all PDFs
    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(uploaded_file.read())

            pdf = PdfReader(temp_file.name)

            for page in pdf.pages:
                full_text += page.extract_text()

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(full_text)

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

    # LLM
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile"
    )

    # Chat input
    query = st.chat_input(
        "Ask Question From PDFs"
    )

    if query:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )

        # Query embedding
        query_embedding = embedding_model.encode([query])

        # Search
        distances, indices = index.search(
            np.array(query_embedding),
            2
        )

        retrieved_text = ""

        for i in indices[0]:
            retrieved_text += chunks[i] + "\n\n"

        # History
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

        Give answer only from context.
        """

        # AI response
        response = llm.invoke(prompt)

        answer = response.content

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    # Display chat
    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.write(msg["content"])

else:

    st.info("Please upload PDF files.")