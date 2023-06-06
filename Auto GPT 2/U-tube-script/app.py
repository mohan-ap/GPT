import os
from apikey import apikey
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,SimpleSequentialChain,SequentialChain
from langchain.memory import ConversationBufferMemory


os.environ['OPENAI_API_KEY']=apikey

st.title('Youtube GPT Title and Script Generator')
prompt=st.text_input('give your prompt here')

memory=ConversationBufferMemory(input_key='topic',memory_key='chat history')

title_template=PromptTemplate(
    input_variables=['topic'],
    template='write me a youtube video title about {topic}'
)

script_template=PromptTemplate(
    input_variables=['title'],
    template='write me a youtube video script based on this title: {title}'
)

llm=OpenAI(temperature=0.9,model="gpt-3.5-turbo")
title_chain=LLMChain(llm=llm,prompt=title_template,verbose=True,output_key='title',memory=memory)
script_chain=LLMChain(llm=llm,prompt=script_template,verbose=True,output_key='script',memory=memory)
sequential_chain=SequentialChain(chains=[title_chain,script_chain],input_variables=['topic'],
                                output_variables=['title','script'],verbose=True)

if prompt:
    response=sequential_chain({'topic':prompt})
    st.write(response['title'])
    st.write(response['script'])

    with st.expander('message history'):
        st.info(memory.buffer)