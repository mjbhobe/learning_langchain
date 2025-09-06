"""
06_RAG_2.py: building a multi document RAG application with LangChain & LangGraph
    In this example we'll use multiple text files, but the code to build the vector
    store is generic and can be used for multiple types of files (PDF, text, markdown,
    Word & Excel files - all saved in 1 location)

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
from collections import defaultdict

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document

# import all applicable loaders
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader,  # for legacy .doc
    UnstructuredExcelLoader,  # for .xlsx/.xls (requires `unstructured` + `openpyxl`)
)
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
faiss_store = pathlib.Path(__file__).parent / "faiss_index_rag2"
# this location can contain files of multiple types (such as txt, md, pdf, docx, xlsx)
doc_store = pathlib.Path(__file__).parent / "news_articles"


# ---- Helper functions --------------------------
def load_all_docs(root: pathlib.Path) -> List:
    """Load documents of multiple types with appropriate loaders."""
    docs = []
    len1 = 0

    # *.txt & *.md - text files & markdown files
    for pattern in ("*.txt", "*.md"):
        for p in root.rglob(pattern):
            docs.extend(TextLoader(str(p), encoding="utf-8").load())
    len2 = len(docs)
    console.print(f"   [green]Loaded {len2} text (*.txt & *.md) files[/green]")
    len1 = len2

    # .pdf (page-aware)
    for p in root.rglob("*.pdf"):
        docs.extend(PyPDFLoader(str(p)).load())
    len2 = len(docs)
    console.print(f"   [green]Loaded {len2-len1} PDF (*.pdf) files[/green]")
    len1 = len2

    # .docx
    for p in root.rglob("*.docx"):
        docs.extend(Docx2txtLoader(str(p)).load())
    len2 = len(docs)
    console.print(f"   [green]Loaded {len2-len1} DOCX (*.docx) files[/green]")
    len1 = len2

    # .doc (legacy; needs unstructured & dependencies)
    for p in root.rglob("*.doc"):
        docs.extend(UnstructuredWordDocumentLoader(str(p)).load())
    len2 = len(docs)
    console.print(f"   [green]Loaded {len2-len1} DOC (*.doc) files[/green]")
    len1 = len2

    # .xlsx / .xls (sheet-aware; needs unstructured + openpyxl)
    for pattern in ("*.xlsx", "*.xls"):
        for p in root.rglob(pattern):
            # strategy="hi_res" optional; leave default for speed
            docs.extend(UnstructuredExcelLoader(str(p)).load())
    len2 = len(docs)
    console.print(
        f"   [green]Loaded {len2-len1} Excel files (*.xlsx/*.xls) files[/green]"
    )
    len1 = len2

    return docs


def normalize_metadata(chunks: List) -> None:
    """Add consistent metadata: source_path, source_file, file_uri, page (int), sheet (str)."""
    for d in chunks:
        src_path = d.metadata.get("source") or d.metadata.get("source_path")
        if not src_path and hasattr(d, "metadata"):
            # some loaders store "filename" or similar
            src_path = d.metadata.get("filename") or d.metadata.get("file_path")

        if src_path:
            p = pathlib.Path(src_path)
            d.metadata["source_path"] = str(p.resolve())
            d.metadata["source_file"] = p.name
            try:
                d.metadata["file_uri"] = p.resolve().as_uri()  # file:// URI
            except Exception:
                d.metadata["file_uri"] = str(p.resolve())
        else:
            d.metadata.setdefault("source_file", "unknown")
            d.metadata.setdefault("source_path", "unknown")

        # Normalize page (for PDFs; sometimes 'page_number')
        page = d.metadata.get("page", d.metadata.get("page_number"))
        if page is not None:
            try:
                d.metadata["page"] = int(page)
            except Exception:
                # keep as-is if not castable
                d.metadata["page"] = page

        # Normalize sheet (for Excel; some loaders use 'sheet' or 'sheet_name')
        sheet = (
            d.metadata.get("sheet")
            or d.metadata.get("sheet_name")
            or d.metadata.get("Sheet")
            or d.metadata.get("worksheet")
        )
        if sheet is not None:
            d.metadata["sheet"] = str(sheet)


def format_references(docs: List[Document]) -> str:
    """Build a stable References block from retrieved docs' metadata."""
    per_file_pages = defaultdict(set)
    file_to_uri = {}

    for d in docs:
        meta = d.metadata or {}
        src_file = (
            meta.get("source_file")
            or meta.get("source_path")
            or meta.get("source")
            or "unknown"
        )
        page = meta.get("page")
        if page is not None:
            # Cast to int when possible to avoid mixed sort
            try:
                page = int(page)
            except Exception:
                pass
        per_file_pages[src_file].add(page)
        # keep the last seen uri (or the first; doesn’t matter much)
        file_to_uri[src_file] = meta.get("file_uri") or meta.get("source_path") or ""

    lines = []
    for src_file in sorted(per_file_pages.keys()):
        pages = per_file_pages[src_file]
        # filter out None pages (text/word may not have pages)
        numeric = sorted([p for p in pages if isinstance(p, int)])
        non_numeric = sorted(
            [p for p in pages if not isinstance(p, int) and p is not None]
        )

        if numeric and non_numeric:
            page_str = f"pages {', '.join(map(str, numeric))}; {', '.join(map(str, non_numeric))}"
        elif numeric:
            page_str = f"pages {', '.join(map(str, numeric))}"
        elif non_numeric:
            page_str = ", ".join(map(str, non_numeric))
        else:
            page_str = "(no page info)"

        uri = file_to_uri.get(src_file) or ""
        if uri:
            lines.append(f"- {src_file} — {page_str} — {uri}")
        else:
            lines.append(f"- {src_file} — {page_str}")

    if not lines:
        return "References: (none)"
    return "References:\n" + "\n".join(lines)


def build_model_context(docs: List[Document]) -> str:
    """builds context for model (from similarity search) with meta data for printing references"""
    tagged_chunks = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src_file = (
            meta.get("source_file")
            or meta.get("source_path")
            or meta.get("source")
            or "unknown"
        )
        page = meta.get("page", "NA")
        tagged_chunks.append(
            f"<<chunk {i} | file:{src_file} | page:{page}>>\n{d.page_content}\n<<end chunk {i}>>"
        )
    return "\n\n".join(tagged_chunks)


def create_or_load_embeddings():
    """creates if not available or loads from disk a FAISS embedding"""
    if not faiss_store.exists():
        console.print(
            f"[yellow]Loading document from URL {doc_store}. Please wait...[/yellow]"
        )
        all_docs = load_all_docs(pathlib.Path(doc_store))
        console.print(
            f"   [blue]Loaded {len(all_docs)} documents from {doc_store}[/blue]"
        )

        # split document into chunks of 1000 chars with 200 chars overlap
        console.print(f"[yellow]Chunking the documents...[/yellow]", end="")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""],
            # add_start_index=True,  # track index in original documen
        )
        all_splits = text_splitter.split_documents(all_docs)
        console.print(f"[blue]Done![/blue]")

        console.print(f"[yellow]Normalizing metadata...[/yellow]", end="")
        normalize_metadata(all_splits)
        console.print(f"[blue]Done![/blue]")

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
            str(faiss_store),
            embeddings,
            allow_dangerous_deserialization=True,
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
            "Based on the following context\n\nContext: {context}\n\n"
            "Answer this question: {question}\n\n"
            "Return only the answer.",
        ),
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
    # retrieve top-k chunks using similarity search (k=6)
    results = vector_store.similarity_search(state["question"], k=6)
    return {"context": results}


def generate(state: State):
    # context = "\n\n".join(doc.page_content for doc in state["context"])
    context = build_model_context(state["context"])
    prompt = prompt_template.invoke({"context": context, "question": state["question"]})
    response = llm.invoke(prompt)

    # and the references
    refs = format_references(state["context"])
    answer = f"{response.content}\n\n{refs}"
    return {"answer": answer}


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
