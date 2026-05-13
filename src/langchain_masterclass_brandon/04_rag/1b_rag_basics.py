"""
1b_rag_basics.py - retrieve semantically similar documents from vector store,
    which we created in 1a_rag_basics.py. NOTE: we do use an LLM in this module!

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

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# load API keys
load_dotenv(override=True)
console = Console()
logger = get_logger()

chroma_index_path = Path(__file__).parent / "chroma_db/books"

if not chroma_index_path.exists():
    # create the embeddings
    logger.fatal(
        f"FATAL ERROR: ChromaDB vector store not found! Expected at {str(chroma_index_path)}"
    )
    console.print(
        f"[red]FATAL ERROR: ChromaDB vector store not found! Expected at {str(chroma_index_path)}[/red]"
    )
    console.print(
        "Please run 1a_rag_basics.py module to create the store BEFORE running this module"
    )
    sys.exit(-1)
else:
    logger.info(f"Loading ChromaDB index from {str(chroma_index_path)}")

    # Initialize embeddings
    embeddings = OpenAIEmbeddings()

    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(chroma_index_path),
    )

    # Initialize retriever (retrieve 3 nearest semantically similar chunks)
    retriever = vector_store.as_retriever(
        # Only returns docs whose similarity score meets a
        # minimum cutoff (what you're using)
        search_type="similarity_score_threshold",
        # k:3 -> return max 3 docs;
        # score_threshold:0.4 ->  the min cosine similarity score
        # (range 0–1) a doc must have to be included in results.
        search_kwargs={"k": 3, "score_threshold": 0.4},
    )

    # infinite loop - simulate a chatbot
    question = ""
    while True:
        console.print("[green]Your question? [/green]", end="")
        question = input().lower().strip()
        if question in ["exit", "quit", "bye"]:
            console.print("[red]Exiting...[/red]")
            break

        # ask the vector store to retrieve semantically similar docs
        relevant_docs = retriever.invoke(question)
        console.print("[yellow]Relevant documents:[/yellow]\n")
        for doc in relevant_docs:
            console.print(f"[blue]{'-*-'*20}[/blue]")
            console.print(Markdown(doc.page_content))
            console.print("\n")

        # run your chain
        # logger.debug(f"Asking LLM to respond to {question}")
        # #result = qa_chain.invoke({"query": question})
        # console.print("[yellow]Answer:[/yellow]\n")
        # console.print(Markdown(result["result"]))
