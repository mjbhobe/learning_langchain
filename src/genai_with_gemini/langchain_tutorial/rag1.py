"""
Illustrates the basic RAG application
"""

import os
import pathlib
from dotenv import load_dotenv, find_dotenv

# import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# the following modules are useful only if you run code in Notebooks
from IPython.display import display
from IPython.display import Markdown

# load env variables from .env file
_ = load_dotenv(find_dotenv())  # read local .env file
# genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


# create instance of the text model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
)

import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

# load contents of an online-blog
print("-" * 80)
print("Loading web-page...", flush=True)
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()


print("-" * 80)
print("Splitting content of web-page...", flush=True)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# index chunks
# save into embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

print("Creating embeddings...", flush=True)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS(embedding_function=embeddings)
_ = vector_store.add_documents(documents=all_splits)

# define the prompts
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = model.invoke(messages)
    return {"answer": response.content}


# build the retrieval graph
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# now ask a question off the webpage
response = graph.invoke({"question": "What is Task Decomposition?"})
print(response["answer"])
