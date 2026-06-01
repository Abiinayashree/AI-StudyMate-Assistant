import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import faiss
import numpy as np
import tempfile

# Load environment variables
load_dotenv()

# Initialize chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Title
st.title("AI StudyMate Assistant")

# Sidebar
with st.sidebar:

    st.title("📚 AI StudyMate Assistant")

    st.subheader("Features")

    st.write("✅ Multi PDF Upload")
    st.write("✅ Semantic Search")
    st.write("✅ AI Question Answering")
    st.write("✅ Conversational Memory")

    st.subheader("System Status")

    st.success("PDF Processing Ready")
    st.success("Embeddings Generated")
    st.success("Vector Database Active")
    st.success("AI Assistant Ready")

    st.subheader("Chat Statistics")

    total_messages = len(st.session_state.messages)

    user_messages = len(
        [
            msg for msg in st.session_state.messages
            if msg["role"] == "user"
        ]
    )

    assistant_messages = len(
        [
            msg for msg in st.session_state.messages
            if msg["role"] == "assistant"
        ]
    )

    st.write(f"Total Messages: {total_messages}")
    st.write(f"User Questions: {user_messages}")
    st.write(f"AI Responses: {assistant_messages}")

    st.caption(
        "AI-Powered Multi-PDF Conversational RAG Assistant"
    )

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type="pdf",
    accept_multiple_files=True
)

# Clear Conversation
if st.button("Clear Conversation"):

    st.session_state.messages = []

    st.rerun()

# Process PDFs
if uploaded_files:

    st.subheader("📂 Uploaded Files")

    for file in uploaded_files:
        st.write(file.name)

    full_text = ""

    # Read PDFs
    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(uploaded_file.read())

            pdf = PdfReader(temp_file.name)

            for page in pdf.pages:

                text = page.extract_text()

                if text:
                    full_text += text

    # Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(full_text)

    st.success(
        f"Total PDFs Uploaded: {len(uploaded_files)}"
    )

    st.success(
        f"Total Chunks Created: {len(chunks)}"
    )

    # Embeddings
    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    embeddings = embedding_model.encode(chunks)

    embedding_array = np.array(
        embeddings,
        dtype=np.float32
    )

    # FAISS
    index = faiss.IndexFlatL2(
        embedding_array.shape[1]
    )

    index.add(embedding_array)

    # Groq LLM
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile"
    )

    # Chat Input
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
        query_embedding = embedding_model.encode(
            [query]
        )

        query_embedding = np.array(
            query_embedding,
            dtype=np.float32
        )

        # Search
        distances, indices = index.search(
            query_embedding,
            2
        )

        retrieved_text = ""

        for i in indices[0]:
            retrieved_text += chunks[i] + "\n\n"

        # History
        history = ""

        for msg in st.session_state.messages:

            history += (
                f"{msg['role']}: "
                f"{msg['content']}\n"
            )

        # Prompt
        prompt = f"""
You are an AI Study Assistant.

Previous Conversation:
{history}

Context:
{retrieved_text}

User Question:
{query}

Answer only from the provided context.
"""

        response = llm.invoke(prompt)

        answer = response.content

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )
        source_text = (
            "📄 Retrieved Context:\n\n"
            + retrieved_text
        )


    # Display Chat
    for msg in st.session_state.messages:

        with st.chat_message(
            msg["role"]
        ):
            st.write(msg["content"])

else:

    st.info(
        "Please upload PDF files."
    )