"""
03_semantic_search.py: Build a semantic search engine over a
    PDF with document loaders, embedding models, and vector stores.

@author: Manish Bhobé
My experiments with AI/Gen AI. Code shared for learning purposes only.
Use at your own risk!!
"""

import pathlib
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown

from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# since we are using Gemini, we'll use Google embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# load API keys from .env files
load_dotenv(override=True)
# for colorful text output
console = Console()

# create our LLM - we'll be using Gemini-2.5-flash
llm = init_chat_model("google_genai:gemini-2.5-flash", temperature=0.0)
faiss_store = pathlib.Path(__file__).parent / "faiss_index"


def create_or_load_embeddings():
    """creates if not available or loads from disk a FAISS embedding"""
    if not faiss_store.exists():
        # load the PDF into memory
        pdf_path = pathlib.Path(__file__).parent / "docs" / "nike-10k-2023.pdf"
        console.print(
            f"[yellow]Loading the PDF {str(pdf_path)}. Please wait...[/yellow]"
        )
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        console.print(f"[blue]Loaded {len(docs)} documents[/blue]")
        console.print(f"[blue]Metadata of first document: {docs[0].metadata}[/blue]")
        console.print(
            f"[blue]First 200 chars of first document: {docs[0].page_content[:200]}[/blue]"
        )

        # split PDF into chunks of 1000 chars with 200 chars overlap
        console.print(f"[yellow]Chunking the PDF. Please wait...[/yellow]")
        chunk_size: int = 1000
        overlap: int = 200

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
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
            "respond from the context only. If answer does not appear in the context respond "
            "with an appropriate polite message, such as \"I'm sorry, I don't have that information.\"",
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
    console.print(f"[yellow]Your query? [/yellow]", end="")
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
    results = vector_store.similarity_search(query)
    console.print(f"[blue]Query: [/blue]{query}")
    console.print(f"[green]Found {len(results)} results [/green]")
    # display the similarity search results
    context = ""
    for i, result in enumerate(results):
        console.print(Markdown(f"**Answer #{i+1}**: {result.page_content}"))
        console.print(f"[blue]Metadata: {result.metadata}\n[/blue]")
        context += f"\n\n{result.page_content}"

    prompt = prompt_template.invoke({"context": context, "question": query})
    response = llm.invoke(prompt)
    md = Markdown(response.content)
    console.print(f"[green]AI: [/green]\n\n")
    console.print(md)
