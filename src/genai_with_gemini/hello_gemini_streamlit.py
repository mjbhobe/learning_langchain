"""hello_gemini_streamlit.py:
This is a streamlit based Q&A application with Google Gemini LLM

@author: Manish Bhobé
My experiments with Python, Data Science, Deep Learning & Generative AI.
Code is made available as-is & for learning purposes only, please use at your own risk!!
"""

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


# create an instance of Gemini text model
# alternatively, use "gemini-pro" instead of "gemini-1.5-flash"
model = genai.GenerativeModel("gemini-pro")


def to_markdown(text):
    """function to convert response from model to correct format"""
    text = text.replace("•", "  *")
    return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))


def get_gemini_response(
    prompt_text: str,
    llm=model,
    temperature=1.0,
    top_p=None,
    top_k=None,
):
    """return response form LLM given prompt_text"""
    try:
        complete_prompt = f"""
        You are adept at providing accurate information on historical events, facts & figures, including 
        dates and times. When an event in the question asked does not occur on date/time asked in the 
        question, please respond with the nearest correct date and time when the event actually occured. 
        To explain any mathematical or statistical concepts, use math equations liberally. When 
        rendering math equations or expressions use double dollar signs to render the latex.
        Always assume that the user is new to the topic, unless explicitly mentioned and provide as detailed
        as response as possible.
        
        {prompt_text}
        """
        gen_config = {"temperature": temperature, "top_k": top_k, "top_p": top_p}
        response = llm.generate_content(complete_prompt, generation_config=gen_config)
        return response.text
    except ValueError as err:
        print(f"get_gemini_response() Error: {err}")
        print(f"{response.prompt_feedback}")


def main():
    # setup Streamlit UI
    st.set_page_config(page_title="Ask Gemini: Q&A Demo with Google Gemini")
    st.header("Q&A with Google Gemini")

    prompt_text = st.text_area("Your Question? ", key="prompt_text")
    submit = st.button("Go!")

    if submit:
        # the submit button is clicked
        response = get_gemini_response(prompt_text, temperature=0.0)
        st.write(response)


if __name__ == "__main__":
    main()
