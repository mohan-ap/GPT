from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain import OpenAI
import glob
import os


os.environ["OPENAI_API_KEY"] = "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"


llm=OpenAI()

def summarize_pdfs_from_folder(pdf_folder_path):
    summaries = []
    for pdf_file in glob.glob(os.path.join(pdf_folder_path, "*.pdf")):
        loader = PyPDFLoader(pdf_file)
        docs = loader.load_and_split()
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)
        print("Summary for: ", pdf_file)
        print(summary)
        print("\n")
        summaries.append(summary)
    
    return summaries

pdf_folder_path = os.path.expanduser("~/Downloads/PDF")
summaries = summarize_pdfs_from_folder(pdf_folder_path)

with open("summaries.txt", "w") as f:
    for summary in summaries:
        f.write(summary + "\n")
