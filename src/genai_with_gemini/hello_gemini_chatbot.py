""" hello_gemini_chatbot.py: Chatbot with Google Gemini
Google Gemini powered Q&A Chatbot with a streamlit GUI

@author: Manish Bhobe
My experiments with Python, Data Science, Deep Learning & Generative AI.
Code is made available as-is & for learning purposes only, please use at your own risk!!
"""

import pathlib
import textwrap
import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st

import google.generativeai as genai

# the following modules are useful only if you run code in Notebooks
from IPython.display import display
from IPython.display import Markdown

# load env variables from .env file
_ = load_dotenv(find_dotenv())  # read local .env file
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


# create instance of the text model
model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat(history=[])  # start with blank history


def to_markdown(text):
    """function to convert response from model to correct format"""
    text = text.replace("•", "  *")
    return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))


def get_gemini_response(prompt_text: str, chat_model=chat, stream=False):
    """return response form LLM given prompt_text"""
    try:
        response = chat_model.send_message(prompt_text, stream=stream)
        return response
    except ValueError as err:
        print(f"get_gemini_response() Error: {err}")
        print(f"{response.prompt_feedback}")


def main():
    # setup Streamlit UI
    st.set_page_config(page_title="Gemini Chatbot")
    st.header("Google Gemini Chatbot 💬")

    # check if we have saved chat history in session, else create
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # create the GUI
    input = st.text_area("Input:", key="input")
    submit = st.button("Ask the question")

    if submit and input:
        response = get_gemini_response(input, stream=True)
        # add the user-query & response to session history
        st.session_state["chat_history"].append(("You", input))
        st.subheader("The response is")
        # we have streamed the response
        response_text = ""
        for chunk in response:
            st.write(chunk.text)
            response_text += chunk.text
        st.session_state["chat_history"].append(("Bot", response_text))
        # display the chat history
        st.subheader("The chat history")
        for role, text in st.session_state["chat_history"]:
            st.write(f"{role}:{text}")


if __name__ == "__main__":
    main()
