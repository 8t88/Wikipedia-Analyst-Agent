from typing import List

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.document_loaders import WikipediaLoader
# from langchain_community.tools.openai_dalle_image_generation import (
#    OpenAIDALLEImageGenerationTool
# )
#from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
import os
import pandas as pd
from langchain.tools import tool
#from langchain.tools.render import format_tool_to_openai_function
#from langchain.schema import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
#from langgraph.prebuilt import create_react_agent
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessageChunk, SystemMessage
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_ollama import ChatOllama

from langchain.agents import create_agent


@tool
def get_wikipedia_docs(query:str, docs_load: int=3) -> List[Document]:
  """
  Tool that takes in a query and returns a list of
  the top 3 wikipedia documents most relevant to that query

  Args:
    query: the question the user is trying to answer
  Returns:
    a list of the top 3 wikipedia documents most relevant to the query
  """

  wikipedia_docs = WikipediaLoader(query, load_max_docs=docs_load)
  return wikipedia_docs.load()

@tool
def get_wiki_tables(wikipedia_docs: List[Document]) -> List[pd.DataFrame]:
  """Tool that takes in a list of wikipedia docs, parses out the url
  from each doc, and extracts any tables from that url.

  Args:
    wikipedia_docs: the list of wikipedia docs; each doc contains the url of the wikipedia page
  Returns:
    a list of dataframes
  """
  links = [x.metadata['source'] for x in wikipedia_docs]
  headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
  tables_pds = []
  for url in links:
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        tables_pds += pd.read_html(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the page: {e}")
    except ValueError as e:
        print(f"Error parsing HTML: {e}")
    return tables_pds


def initialize_agent():
    
    llm = ChatOllama(model="llama3.1:8b", format="json", temperature=0)
    
    memory = MemorySaver()
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    retriever = WikipediaRetriever(load_max_docs=10)

    tools = [wikipedia, get_wikipedia_docs, get_wiki_tables]

    system_prompt = """
    You are a well-educated research assistant trying to find insights for a user based on data pulled from wikipedia.
    """
    
    agent = create_agent(
        llm,
        tools,
        system_prompt=system_prompt,
    )

    return agent