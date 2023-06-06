# from langchain.agents import create_csv_agent
# from langchain.llms import OpenAI
# import os

# os.environ["OPENAI_API_KEY"] = "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"



# root_dir = os.path.expanduser("~/Downloads/")
# file_path = os.path.join(root_dir, "organizations-100.csv")

# agent = create_csv_agent(OpenAI(temperature=0), file_path, verbose=True)
# query=input("ask your question ")
# print(agent.run(query))



import streamlit as st
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
import os

os.environ["OPENAI_API_KEY"] = "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"

st.title("CSV Query Executor")

uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    file_path = os.path.join(os.getcwd(), uploaded_file.name)  #cwd-current working directory

    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    agent = create_csv_agent(OpenAI(temperature=0), file_path, verbose=True)
    query = st.text_input("Ask your question:")
    if query:
        result = agent.run(query)
        st.write(result)
