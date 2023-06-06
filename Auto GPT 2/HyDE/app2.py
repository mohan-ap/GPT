from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI



os.environ["OPENAI_API_KEY"] = "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"

root_dir = os.path.expanduser("~/Downloads/")
pdf_path = os.path.join(root_dir, "Stephen Hawking - A Brief History Of Time.pdf")

reader = PdfReader(pdf_path)
# print(reader)

raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

# print(raw_text)
# print(raw_text[:100])

text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 100,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)
# print(len(texts))
# print(texts[0])

embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)


chain =load_qa_chain(OpenAI(), chain_type="stuff")
query = input("Enter your question: ")
docs = docsearch.similarity_search(query)

print(docs)
answer = chain.run(input_documents=docs, question=query)
print("Answer:", answer)
