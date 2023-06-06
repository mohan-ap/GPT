import streamlit as st
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
import textwrap
from apikey import apikey
import os

os.environ['OPENAI_API_KEY'] = apikey

embeddings = OpenAIEmbeddings()


def create_db_youtube_url(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.2)
    template = """
        You are a helpful assistant that can answer questions about YouTube videos
        based on the video's transcript: {docs}

        Only use the factual information from the transcript to answer the question.

        If you feel like you don't have enough information to answer the question, say "I don't know".

        Your answers should be verbose and detailed.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("/n", "")
    return response


def main():
    st.title("YouTube Video Assistant")
    st.write("Ask questions related to a YouTube video's transcript.")

    video_url = st.text_input("Enter YouTube video URL")
    query = st.text_input("Ask a question")
    if st.button("Get Answer"):
        if video_url and query:
            db = create_db_youtube_url(video_url)
            response = get_response_from_query(db, query)
            response = textwrap.fill(response, width=200)
            st.write("Answer:")
            st.write(response)
        else:
            st.write("Please enter both the YouTube video URL and a question.")


if __name__ == "__main__":
    main()
