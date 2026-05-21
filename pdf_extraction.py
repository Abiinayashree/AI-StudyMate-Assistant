import streamlit as st
from pypdf import PdfReader

st.title("AI StudyMate Assistant")

st.subheader("Educational PDF Processing System")

st.write("Uploaded PDF content extraction using PyPDF.")

# Read PDF
pdf = PdfReader("documents/sample.pdf")

# Store extracted text
text = ""

# Loop through pages
for page in pdf.pages:
    text += page.extract_text()

# Show extracted text
st.text_area("Extracted PDF Content", text, height=400)