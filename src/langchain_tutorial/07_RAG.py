import bs4, sys
import pathlib
from dotenv import load_dotenv
from typing import List, TypedDict
from rich.console import Console
from rich.markdown import Markdown

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# since we are using Gemini, we'll use Google embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from langgraph.graph import StateGraph, START, END

# load API keys from .env files
load_dotenv(override=True)
# for colorful text output
console = Console()

#### INDEXING ####

# Load blog
import bs4
from langchain_community.document_loaders import WebBaseLoader
console.print("[yellow]Creating vector store. Please wait...[/yellow]")
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
blog_docs = loader.load()

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)

# Make splits
splits = text_splitter.split_documents(blog_docs)

# Index
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Use a Gemini embedding model that is suitable for retrieval.
# It is important to match the model to the task.
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="retrieval_document")
vectorstore = FAISS.from_documents(
    documents=splits, 
    embedding=embeddings
)

retriever = vectorstore.as_retriever()


# now ask the LLM to respond
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Multi Query: Different Perspectives
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_perspectives 
    # Swap out ChatOpenAI for the Gemini 2.5 Flash model
    | ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0) 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

from langchain.load import dumps, loads

def get_unique_union(documents: list[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]

# Retrieve
question = "What is task decomposition for LLM agents?"
retrieval_chain = generate_queries | retriever.map() | get_unique_union
docs = retrieval_chain.invoke({"question":question})
print(f"Got {len(docs)} documents")

from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0) 

final_rag_chain = (
    {"context": retrieval_chain, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)

response = final_rag_chain.invoke({"question":question})
console.print(response)