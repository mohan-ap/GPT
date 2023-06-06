# from langchain.agents import create_sql_agent
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# from langchain.sql_database import SQLDatabase
# from langchain.llms.openai import OpenAI
# from langchain.agents import AgentExecutor
# import os
# from langchain import HuggingFaceHub

# os.environ["OPENAI_API_KEY"] = "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"

# db = SQLDatabase.from_uri("sqlite:///database.db")

# toolkit = SQLDatabaseToolkit(db=db, llm=HuggingFaceHub(repo_id="google/flan-t5-xl", 
#                                         huggingfacehub_api_token="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",
#                                         model_kwargs={"temperature":0, 
#                                                       "max_length":64}))

# agent_executor = create_sql_agent(
#     llm=toolkit.llm,
#     toolkit=toolkit,
#     verbose=True
# )
# query=input("ask your question ")
# print(agent_executor.run(query))

import streamlit as st
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor
import os

os.environ["OPENAI_API_KEY"] = "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"

def create_db_connection(db_file_path):
    return SQLDatabase.from_uri(f"sqlite:///{db_file_path}")

def main():
    st.title("SQL Query Executor")

    uploaded_file = st.file_uploader("Upload database file", type=["db", "sqlite", "sqlite3"])
    query = st.text_input("Ask your question")

    if st.button("Execute") and uploaded_file is not None:
        file_name = uploaded_file.name
        file_path = os.path.join(os.getcwd(), file_name) 
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        db = create_db_connection(file_path)
        toolkit = SQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))
        agent_executor = create_sql_agent(llm=toolkit.llm, toolkit=toolkit, verbose=True)

        result = agent_executor.run(query)
        st.write(result)

if __name__ == "__main__":
    main()


