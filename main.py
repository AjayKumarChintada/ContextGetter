import nest_asyncio

nest_asyncio.apply()
from dotenv import load_dotenv
import os
import openai
import logging
import sys


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_API_KEY')

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI

from llama_index.core import VectorStoreIndex,SimpleDirectoryReader
from llama_index.core.query_engine import RetrieverQueryEngine




documents=SimpleDirectoryReader("soptest").load_data()


# initialize LLM + node parser
llm = OpenAI(model="gpt-4")
splitter = SentenceSplitter(chunk_size=1024)

nodes = splitter.get_nodes_from_documents(documents,show_progress=True)

storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

INDEX = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
)



def rag_context(question):
    vector_retriever = VectorIndexRetriever(INDEX,similarity_top_k=10)
    response = vector_retriever.retrieve(question)
    resp = [{"context": resp.get_text(), "similarity score": resp.get_score()} for resp in response]
    print(resp)
    return resp

print(rag_context("tell about cuts and wounds?"))
