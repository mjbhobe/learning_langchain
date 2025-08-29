"""
06_RAG_1.py: building a naive RAG application with Langchain & LangGraph

@author: Manish Bhobé
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

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

# create our LLM - we'll be using Gemini-2.5-flash
llm = init_chat_model("google_genai:gemini-2.5-flash", temperature=0.0)
faiss_store = pathlib.Path(__file__).parent / "faiss_index_rag1"


def create_or_load_embeddings():
    """creates if not available or loads from disk a FAISS embedding"""
    if not faiss_store.exists():
        # in this example we'll load document from a URL
        web_url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
        console.print(
            f"[yellow]Loading document from URL {web_url}. Please wait...[/yellow]"
        )
        loader = WebBaseLoader(
            web_paths=(web_url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()

        console.print(f"[blue]Loaded {len(docs)} documents from URL[/blue]")
        console.print(f"[blue]Metadata of first document: {docs[0].metadata}[/blue]")
        console.print(
            f"[blue]First 200 chars of first document: {docs[0].page_content[:200]}[/blue]"
        )

        # split document into chunks of 1000 chars with 200 chars overlap
        console.print(f"[yellow]Chunking the PDF. Please wait...[/yellow]")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,  # track index in original documen
        )
        all_splits = text_splitter.split_documents(docs)
        console.print(f"[blue]Created {len(all_splits)} chunks[/blue]")

        # save to embeddings

        console.print("[yellow]Creating embeddings. Please wait...[/yellow]")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(all_splits, embeddings)
        vector_store.save_local(str(faiss_store))
        console.print(
            f"[yellow]Local embeddings created at {str(faiss_store)}[/yellow]"
        )
    else:
        console.print(
            f"[yellow]Loading existing embeddings from {str(faiss_store)}[/yellow]"
        )
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.load_local(
            str(faiss_store), embeddings, allow_dangerous_deserialization=True
        )
    return vector_store


vector_store = create_or_load_embeddings()

# now ask the LLM to respond
from langchain_core.prompts import ChatPromptTemplate

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Read the question and scan the context provided and "
            "respond only from the context provided. If answer does not appear in the context respond "
            "with an appropriate polite message, such as \"I'm sorry, I don't have that information.\"",
        ),
        (
            "user",
            "Based on the following context\n\nContext: {context},\n\nanswer this question: {question}",
        ),
        # ("assistant", "Sure, let me think..."),
    ],
)


# now build a graph using LangGraph
# NOTE: this does not store the query history!!
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# define the nodes of the graph
def retrieve(state: State):
    results = vector_store.similarity_search(state["question"])
    return {"context": results}


def generate(state: State):
    context = "\n\n".join(doc.page_content for doc in state["context"])
    prompt = prompt_template.invoke({"context": context, "question": state["question"]})
    response = llm.invoke(prompt)
    return {"answer": response.content}


builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)
# build the graph
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
graph = builder.compile()

# and ask away

query = ""
while True:
    console.print(f"[blue]Your query? [/blue]", end="")
    query = input().strip().lower()
    if len(query) <= 0:
        # user must eter a query
        console.print("[red]Please enter a query![/red]")
        continue
    elif query in ["exit", "quit", "q", "bye"]:
        # and it should not be one of these words
        console.print("[red]Exiting application. Bye![/red]")
        break

    # get the context for the query from the documents
    response = graph.invoke({"question": query})
    console.print(f"[green]AI Response: [/green]\n")
    console.print(Markdown(response["answer"]))
