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
# I seem to have permanently exhausted rate limt on Google embeddings on free tier,
# I don't want to enable billing, so am switching to Cohere embeddings
# from langchain_cohere import CohereEmbeddings

# since we are using OpenAI we'll use OpenAI embeddings
from langchain_openai import OpenAI, OpenAIEmbeddings

# and we'll be using Chroma embeddings
from langchain_community.vectorstores import Chroma

# load API keys from .env files
load_dotenv(override=True)
# for colorful text output
console = Console()

# llm = init_chat_model("google_genai:gemini-2.5-flash", temperature=0.0)
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", batch_size=65)
# embeddings = CohereEmbeddings(model="embed-english-v3.0")
# vector_store = pathlib.Path(__file__).parent / "faiss_index_gemini_cohere"

# we'll use OpenAI gpt-4o-mini
llm = init_chat_model("gpt-5-nano", model_provider="openai", temperature=0.0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = pathlib.Path(__file__).parent / "chroma_db/chroma_index_openai"
# faiss_store = pathlib.Path(__file__).parent / "faiss_index_openai"


# we'll use Anthropic's Claude Sonnect LLM, Cohere embeddings & FAISS vector DB
# llm = init_chat_model("claude-sonnet-4-5", model_provider="anthropic", temperature=0.0)
# embeddings = CohereEmbeddings(model="embed-english-v3.0")
# vector_store = pathlib.Path(__file__).parent / "faiss_index_anthropic"


def create_or_load_embeddings(
    embeddings, chroma_store, chunk_size=300, chunk_overlap=50
):
    """creates if not available or loads from disk a Chroma DB vector store"""
    if not chroma_store.exists():
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

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        all_splits = text_splitter.split_documents(docs)
        console.print(f"Created {len(all_splits)} chunks")

        # save to embeddings

        console.print("Creating embeddings. Please wait...", style="#C8A16D")
        vector_store = Chroma.from_documents(
            documents=all_splits,
            embedding=embeddings,
            persist_directory=str(chroma_store),
        )
        # retriever = vector_store.as_retriever()
        console.print(
            f"[yellow]Local embeddings created at {str(chroma_store)}[/yellow]"
        )
    else:
        console.print(
            f"Loading existing embeddings from {str(faiss_store)}", style="#C8A16D"
        )
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=str(chroma_store),
        )
        retriever = vector_store.as_retriever()

    return vector_store


vector_store = create_or_load_embeddings(embeddings, vector_store)

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
        console.print(Markdown(f"**Answer #{i + 1}**: {result.page_content}"))
        console.print(f"Metadata: {result.metadata}\n")
        context += f"\n\n{result.page_content}"

    prompt = prompt_template.invoke({"context": context, "question": query})
    response = llm.invoke(prompt)
    md = Markdown(dedent(response.content))
    console.print(f"AI:", style="#85C46C")
    console.print(md)
