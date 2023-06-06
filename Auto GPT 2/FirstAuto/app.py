import os
from langchain.vectorstores import FAISS # you can use Redis and Pinecone 
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.experimental import AutoGPT
from langchain.chat_models import ChatOpenAI
import faiss


os.environ['SERPAPI_API_KEY'] = "2147961026a9bc30e50e184f62d00f9f87408714b51cfade2487d5d89cab13b2"

os.environ['OPENAI_API_KEY'] = "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"

from langchain.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool

search = SerpAPIWrapper()
tools = [
    Tool(
        name = "search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions"
    ),
    WriteFileTool(),
    ReadFileTool(),
]
embeddings_model = OpenAIEmbeddings()

embedding_size = 1536 
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})


agent = AutoGPT.from_llm_and_tools(
    ai_name="Tom",
    ai_role="Assistant",
    tools=tools,
    llm=ChatOpenAI(temperature=0),
    memory=vectorstore.as_retriever()
)

agent.chain.verbose = False

agent.run(["write 5 delicious foods for breakfast in tamilnadu"])

