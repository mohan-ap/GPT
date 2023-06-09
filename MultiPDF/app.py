from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
import os
from langchain.vectorstores import Chroma,FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

os.environ["OPENAI_API_KEY"] = "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"

pdf_folder_path = os.path.expanduser("~/Downloads/PDF")

os.listdir(pdf_folder_path)

loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]

# print(loaders)

index = VectorstoreIndexCreator(
    vectorstore_cls=FAISS, 
    embedding=OpenAIEmbeddings(),
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
).from_loaders(loaders)

#print(index.query('Who is stephen'))
print(index.query_with_sources('who is stephen'))
