"""
semantic_search_engine.py: building a Semantic Search Engine with Google Gemini

Building a RAG framework with document loader, embedding & vector store to search
over a PDF document
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

# we will work on the Nike-10K PDF document in the same folder
from langchain_community.document_loaders import PyPDFLoader

print("Loading PDF. Please wait...", flush=True)

file_path = pathlib.Path(__file__).parent / "nike-10k-2023.pdf"
print(f"{str(file_path)} exists? {file_path.exists()}")
# load the PDF
loader = PyPDFLoader(str(file_path))
docs = loader.load()
print(len(docs))
print(f"Metadata: {docs[0].metadata}")
print(f"First 200 chars: {docs[0].page_content[:200]}\n")

# split the PDF into chunks of 1000 chars with 200 char overlap
print("Chunking PDF. Please wait...", flush=True)
chunk_len: int = 1000
overlap: int = 200

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_len, chunk_overlap=overlap, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
print(f"Split count: {len(all_splits)}")

# save into embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

print("Creating embeddings...", flush=True)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = FAISS.from_documents(all_splits, embeddings)
# vector_store = FAISS(embedding_function=embeddings)
# ids = vector_store.add_documents(documents=all_splits)
print("Done!")

# now let's retrive responses from vector store
print("-" * 80)
q1: str = "How many distribution centers does Nike have in the US"
results = vector_store.similarity_search(q1)
print(f"Q: {q1}")
print(f"A: {results[0]}")

# asynch query (doesn't seem to work!)
# q2: str = "When was Nike Incorporated?"
# results = await vector_store.asimilarity_search(q2)
# print(f"Q: {q2}")
# print(f"A: {results[0]}")

# let's check response & scores
print("-" * 80)
q3: str = "What was Nike's revenue in 2023?"
results = vector_store.similarity_search_with_score(q3)
doc, score = results[0]
print(f"Q: {q3}")
print(f"A: score: {score}\ndocs: {doc}")

# using Retrievers
from typing import List
from langchain_core.documents import Document
from langchain_core.runnables import chain


print("-" * 80)
@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)

results = retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)
print(results)

print("-" * 80)
# we can use a vector-store as a retriever
retriever2 = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k":1},
)
results = retriever2.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)
print(results)
