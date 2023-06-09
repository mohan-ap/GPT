import textwrap
import streamlit as st
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain import OpenAI
import os


os.environ["OPENAI_API_KEY"] = "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"


llm = OpenAI()

def summarize_pdfs_from_folder():
    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True)
    if st.button("Summarize"):
        summaries = []
        for file in uploaded_files:
            pdf_file_path = os.path.join(os.getcwd(), file.name)
            with open(pdf_file_path, "wb") as f:
                f.write(file.getbuffer())
            loader = PyPDFLoader(pdf_file_path)
            docs = loader.load_and_split()
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            summary = chain.run(docs)
            st.write("Summary for: ", file.name)
            st.write(summary)
            summaries.append(summary)

        return summaries
    
st.title("News Documents summary Creator")
summaries = summarize_pdfs_from_folder()

if summaries:
    formatted_summaries = [textwrap.fill(summary, width=100) for summary in summaries]
    formatted_text = "\n".join(formatted_summaries)
    with open("summaries.txt", "w") as f:
        f.write(formatted_text)