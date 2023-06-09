import streamlit as st
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
import os
from langchain.vectorstores import Chroma,FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

os.environ["OPENAI_API_KEY"] = "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"

pdf_folder_path = os.path.join(os.getcwd(), "uploaded_pdfs")

os.makedirs(pdf_folder_path, exist_ok=True)

def upload_pdf():
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        with open(os.path.join(pdf_folder_path, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully!")

def search_index(query):
    index = VectorstoreIndexCreator(
        vectorstore_cls=FAISS, 
        embedding=OpenAIEmbeddings(),
        text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    ).from_loaders(loaders)
    
    results = index.query_with_sources(query)
    
    st.write("Search Results:")
    st.write(results)


def main():
    st.title("Question Answering from Multiple files")

    st.header("Upload PDF files")
    upload_pdf()

    st.header("Ask a question")
    query = st.text_input("Enter your question")
    if st.button("Search"):
        search_index(query)

       

if __name__ == "__main__":
    loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]
    
    main()
