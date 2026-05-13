"""
2b_rag_basics_metadata.py - query the vector store created in 2a_rag_basics_metadata.py
  We just do a vector_store.similarity_search() to retrieve similar chunks & meta-data
  NOTE: we do not call LLM yet!

@author: Manish Bhobé
My experiments with Python, AI/ML and Gen AI.
Code is shared for learning purposed only - use at own risk!
"""

import sys, os, time
from pathlib import Path

# NOTE: I am adding the parent folder of this file to the Python
# sys.path, so I can use utility functions in the utils/rich_logging.py file!
append_to_sys_path = Path(__file__).parent.parent
if str(append_to_sys_path) not in sys.path:
    sys.path.append(str(append_to_sys_path))

from dotenv import load_dotenv

from rich.console import Console
from utils.rich_logging import get_logger

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# load API keys
load_dotenv(override=True)
console = Console()
logger = get_logger()

chromadb_index_path = Path(__file__).parent / "chroma_db" / "books_with_metadata"
source_docs_path = Path(__file__).parent / "books"

if not chromadb_index_path.exists():
    # create the embeddings
    console.print(
        "[yellow]Persistent directory does not exist. Please run 2a_rag_basics_metadata.py to create it[/yellow]"
    )
    sys.exit(-1)

console.print(f"Loading embeddings from {chromadb_index_path}\n")
# load the vector-store into memory
vector_store = Chroma(
    persist_directory=str(chromadb_index_path),
    embedding_function=OpenAIEmbeddings(),
)

# Initialize retriever (retrieve 3 nearest semantically similar chunks)
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.4},
)

# now we will pass in a query & get relevant docs + metadata from vector store
query = ""
while True:
    console.print("[green]Your query? [/green]", end="")
    query = input().strip().lower()
    if query in ["quit", "exit", "bye"]:
        break

    retrieved_docs = retriever.invoke(query)
    console.print("[blue]Retrieved docs:[/blue]\n")
    for i, doc in enumerate(retrieved_docs):
        console.print(f"[yellow]Document #{i:2d}[/yellow]")
        console.print(doc.page_content)
        console.print(f"\n[green]Source:[/green]{doc.metadata['source']}\n")
