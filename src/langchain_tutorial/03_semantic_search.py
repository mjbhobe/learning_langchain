"""
03_semantic_search.py: Build a semantic search engine over a
    PDF with document loaders, embedding models, and vector stores.

@author: Manish Bhobé
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

import pathlib
from textwrap import dedent
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# since we are using Gemini, we'll use Google embeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# since we are using OpenAI we'll use OpenAI embeddings
#from langchain_openai import OpenAIEmbeddings
# since we are using Cohere embeddings
from langchain_cohere import CohereEmbeddings

from langchain_community.vectorstores import FAISS

# load API keys from .env files
load_dotenv(override=True)
# for colorful text output
console = Console()

# we'll use Google Gemini Flash 2.0
# llm = init_chat_model("google_genai:gemini-2.0-flash", temperature=0.0)
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# faiss_store = pathlib.Path(__file__).parent / "faiss_index_gemini"


# we'll use OpenAI gpt-4o-mini
# llm = init_chat_model("gpt-4o-mini", model_provider="openai", temperature=0.0)
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# faiss_store = pathlib.Path(__file__).parent / "faiss_index_openai"


# we'll use Anthropic's Claude Sonnect LLM, Cohere embeddings & FAISS vector DB
llm = init_chat_model("claude-3-7-sonnet-20250219", model_provider="anthropic", temperature=0.0)
embeddings = CohereEmbeddings(model="embed-english-v3.0")
faiss_store = pathlib.Path(__file__).parent / "faiss_index_anthropic"


def create_or_load_embeddings():
    """creates if not available or loads from disk a FAISS embedding"""
    if not faiss_store.exists():
        # load the PDF into memory
        pdf_path = pathlib.Path(__file__).parent / "docs" / "nike-10k-2023.pdf"
        console.print(
            f"Loading the PDF {str(pdf_path)}. Please wait...",
            style="#C8A16D",
        )
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        console.print(f"Loaded {len(docs)} documents", style="#6C95EB")
        console.print(
            f"Metadata of first document: {docs[0].metadata}", style="#6C95EB"
        )
        console.print(
            f"First 200 chars of first document: {docs[0].page_content[:200]}",
            style="#6C95EB",
        )

        # split PDF into chunks of 1000 chars with 200 chars overlap
        console.print(f"Chunking the PDF. Please wait...", style="#C8A16D")
        chunk_size: int = 1000
        overlap: int = 200

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        )
        all_splits = text_splitter.split_documents(docs)
        console.print(f"Created {len(all_splits)} chunks")

        # save to embeddings

        console.print("Creating embeddings. Please wait...", style="#C8A16D")
        # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_documents(all_splits, embeddings)
        vector_store.save_local(str(faiss_store))
        console.print(
            f"Local embeddings created at {str(faiss_store)}",
            style="#C8A16D",
        )
    else:
        console.print(
            f"Loading existing embeddings from {str(faiss_store)}", style="#C8A16D"
        )
        # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
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
            "respond from the context only. If answer does not appear in the context respond "
            "with an appropriate polite message, such as \"I'm sorry, I don't have that information.\"."
            "Don't add any other text to response, such as 'Based on the context provided...' etc.",
        ),
        (
            "user",
            "Based on the following context\n\nContext: {context},\n\nanswer this question: {question}",
        ),
        # ("assistant", "Sure, let me think..."),
    ],
)


query = ""
while True:
    console.print(f"Your query? ", end="", style="#C8A16D")
    query = input().strip().lower()
    if len(query) <= 0:
        # user must enter a query
        console.print("[red]Please enter a query![/red]")
        continue
    elif query in ["exit", "quit", "q", "bye"]:
        # but if it is one of these words, the quit
        console.print("[red]Exiting application. Bye![/red]")
        break

    # get the context for the query from the documents
    results = vector_store.similarity_search(query)
    console.print(f"Query: {query}")
    console.print(f"Found {len(results)} results ", style="#85C46C")
    # display the similarity search results
    context = ""
    for i, result in enumerate(results):
        console.print(Markdown(f"**Answer #{i+1}**: {result.page_content}"))
        console.print(f"Metadata: {result.metadata}\n")
        context += f"\n\n{result.page_content}"

    prompt = prompt_template.invoke({"context": context, "question": query})
    response = llm.invoke(prompt)
    md = Markdown(dedent(response.content))
    console.print(f"AI:", style="#85C46C")
    console.print(md)
