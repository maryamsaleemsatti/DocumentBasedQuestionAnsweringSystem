import streamlit as st
import os 
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from io import BytesIO

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-proj-lRpGsITlhtq1ypFmRYxHT3BlbkFJNboqF55ZRNopWI2lPovT"
theming = "fantastic"
# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    raw_text = ''
    with BytesIO(uploaded_file.read()) as pdf_bytes:
        pdfreader = PdfReader(pdf_bytes)
        for page in pdfreader.pages:
            content = page.extract_text()
            if content:
                raw_text += content
    return raw_text

# Function to search for answer in document
def search_document(query, raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()
    document_search = FAISS.from_texts(texts, embeddings)

    # Search for answer
    docs = document_search.similarity_search(query)
    return docs[0]

# Streamlit app
def main():
    # Background image CSS
    page_bg_img = """
    <style>
    body {
        background-image: url("https://i.postimg.cc/4xgNnkfX/Untitled-design.png");
        background-size: cover;
        background-position: center center;
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.write("---")
    st.sidebar.title("ABOUT")
    st.sidebar.write("---")
    st.sidebar.write("This interface is created by Inshira and Maryam.")
    st.sidebar.write("You can use this to upload a single PDF document of any length and ask questions about its content.")
    st.sidebar.write("---")
    st.sidebar.write("**Note:**")
    st.sidebar.write("- This app was created on May 14, 2024.")
    st.sidebar.write("- It utilizes OpenAI and Langchain technologies for text processing.")
    st.sidebar.write("---")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    query = st.text_input("Enter a question in context to the file:")
    
    # Make the button blue
    if uploaded_file:
        if st.button("Submit Query"):
            raw_text = extract_text_from_pdf(uploaded_file)
            answer = search_document(query, raw_text)
            st.write("Answer:", answer)

if __name__ == "__main__":
    main()
