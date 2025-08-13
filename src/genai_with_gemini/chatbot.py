import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

# for Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# load the API keys from local .env file
_ = load_dotenv(find_dotenv())  # read local .env file

# create the Gemini text model
llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.6,
    convert_system_message_to_human=True,
)

# define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Please respond to the queries in as much detail as possible. Keep a friendly tone",
        ),
        (
            "user",
            "Question: {question}",
        ),
    ]
)

st.title("Langchain demo with Google Gemini")
input_text = st.text_input("Search topic you want")

out_parser = StrOutputParser()
chain = prompt | llm | out_parser

if input_text:
    # user has entered some input
    st.write(chain.invoke({"question": input_text}))
