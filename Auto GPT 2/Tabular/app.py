import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA


os.environ["OPENAI_API_KEY"] = "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"

def create_db():
    root_dir = os.path.expanduser("~/Downloads/")
    file_path = os.path.join(root_dir, "organizations-100.csv")
    loader = CSVLoader (file_path=file_path)
    data=loader.load()
    # print(data)
    embeddings = OpenAIEmbeddings()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    # print(text_splitter)
    docs = text_splitter.split_documents(data)
    # print(docs)
    db = Chroma.from_documents(docs, embeddings)
    # print(db)
    return db

def get_response_from_query(db,query):
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=db.as_retriever())

    response=qa.run(query)
    return response

db=create_db()

query =input("Ask question related to this csv document: ")
response= get_response_from_query(db, query)
print(response)



