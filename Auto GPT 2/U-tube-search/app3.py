from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import(
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
import textwrap
from apikey import apikey
import os

os.environ['OPENAI_API_KEY']=apikey

embeddings=OpenAIEmbeddings()



def create_db_youtube_url(video_url):
    loader=YoutubeLoader.from_youtube_url(video_url)
    transcript=loader.load()
    # print(transcript)
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    docs=text_splitter.split_documents(transcript)
    # print(len(docs))
    db=FAISS.from_documents(docs,embeddings)
    # print(db)
    return db

def get_response_from_query(db,query,k=4): #gpt-3.5-turbo can handle up to 4097 tokens, thats why give as k=4 --4*1000=4000
    docs=db.similarity_search(query,k=k)
    # print(docs)
    docs_page_content = " ".join([d.page_content for d in docs])
    # print(docs_page_content)
    chat=ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0.2)
    template = """
        You are a helpful assistant that can answer questions about youtube videos 
        based on the video's transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
    """
    system_message_prompt=SystemMessagePromptTemplate.from_template(template)

    human_template="answer the following question: {question}"
    human_message_prompt=HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt=ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response=chain.run(question=query,docs=docs_page_content)
    # print(response)
    response=response.replace("/n","")
    # print(response)
    return response



video_url = "https://www.youtube.com/watch?v=L_Guz73e6fw"
db=create_db_youtube_url(video_url)

query =input("Ask question related to this video: ")
response= get_response_from_query(db, query)
print(textwrap.fill(response, width=200))

