from langchain.agents import create_pandas_dataframe_agent

from langchain.llms import OpenAI
import pandas as pd
import os

os.environ["OPENAI_API_KEY"] = "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"

root_dir = os.path.expanduser("~/Downloads/")
file_path = os.path.join(root_dir, "organizations-100.csv")
df = pd.read_csv(file_path)
agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)
query=input("ask your question ")
print(agent.run(query))