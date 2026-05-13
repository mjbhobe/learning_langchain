"""
1a_rag_basics.py - build the vector store for the file books/odyssey.txt.
    The vector store is created in the chroma_db/books subfolder.
    NOTE: we do not use an LLM in this module!

@author: Manish Bhobé
My experiments with Python, AI/ML and Gen AI.
Code is shared for learning purposed only - use at own risk!
"""

import sys
from pathlib import Path

# NOTE: I am adding the parent folder of this file to the Python
# sys.path, so I can use utility functions in the utils/rich_logging.py file!
append_to_sys_path = Path(__file__).parent.parent
if str(append_to_sys_path) not in sys.path:
    sys.path.append(str(append_to_sys_path))

from dotenv import load_dotenv

from rich.console import Console
from rich.markdown import Markdown
from utils.rich_logging import get_logger

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# load API keys
load_dotenv(override=True)
console = Console()
logger = get_logger()

chroma_index_path = Path(__file__).parent / "chroma_db/books"

if not chroma_index_path.exists():
    # create the embeddings
    logger.info("Creating offline ChromaDB vector store")

    # build path to the file we want to embed
    source_docs_path = Path(__file__).parent / "books" / "odyssey.txt"
    if not source_docs_path.exists():
        logger.fatal(f"FATAL ERROR: could not find path {source_docs_path}")
    assert (
        source_docs_path.exists()
    ), f"FATAL ERROR: could not find path {source_docs_path}"

    # load data from source file
    data_loader = TextLoader(str(source_docs_path), encoding="utf-8")
    documents = data_loader.load()
    logger.info(f"[green]Documents:[/green]{documents}")

    # split documents into manageable chunks
    console.print(f"[green]Splitting documents...[/green]")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"[yellow]No of chunks:[/yellow]{len(chunks)}")

    # build vector store & save offline
    console.print(f"[green]Creating embeddings...[/green]")
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(chroma_index_path),
    )
    logger.info(f"Embeddings saved to path {str(chroma_index_path)}")
else:
    logger.info(f"ChromaDB store exists at path {str(chroma_index_path)}")
    console.print(
        f"[green]ChromaDB store exists at path {str(chroma_index_path)}[/green]"
    )
