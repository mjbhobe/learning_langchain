from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

# load API keys from .env file
load_dotenv(override=True)
console = Console()

# create instance of Gemini 3 flash model
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.1,
)

# ask the model a question
prompt = "Tell me a joke about Rahul Gandhi"
response = gemini_llm.invoke(prompt)
console.print("[green]Response from Gemini[/green]")
# console.print(Markdown(response.content))
console.print(response.content)

messages = [
    SystemMessage(content="You're a helpful programming assistant"),
    HumanMessage(content="Write a Python function to calculate factorial"),
]


response = gemini_llm.invoke(messages)
console.print("[green]Code Generation from Gemini[/green]")
# console.print(Markdown(response.content))
console.print(response.content)


# now let's go in a loop
prompt = PromptTemplate.from_template("Tell me a joke about {topic}")
parser = StrOutputParser()
chain = prompt | gemini_llm | parser

for topic in ["Rahul Gandhi", "Virat Kohli", "Langchain"]:
    console.print(f"[green]Tell me a joke about {topic}[/green]")
    console.print(chain.invoke({"topic": topic}))
    console.print("\n")


# a somewhat complex chain, composed of subchains
story_prompt = PromptTemplate.from_template("Write a short story about {topic}")
story_chain = story_prompt | gemini_llm | StrOutputParser()

analysis_prompt = PromptTemplate.from_template("Analyze the following story: {story}")
analysis_chain = analysis_prompt | gemini_llm | StrOutputParser()

combined_chain = story_chain | analysis_chain
response = combined_chain.invoke({"topic": "The impact of Gen AI on Banking"})
console.print(response)

# let's generate an image using Gemini 2.5 Flash Image model
from langchain_google_genai import ChatGoogleGenerativeAI, Modality
from PIL import Image
import base64
from io import BytesIO
import pathlib

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-image",
    response_modalities=[Modality.IMAGE],
    temperature=0.1,
    # Optional: configure aspect ratio or other parameters
    image_config={
        "aspect_ratio": "1:1",
    },
)

image_prompt = "Generate a detailed technical diagram of an AI agent"
response = model.invoke(prompt)
image_path = pathlib.Path(__file__).parent / "generated_image.png"

if response.content:
    # Assuming response.content[0] holds the image data in base64 format (this may need adjustment)
    # Check actual object structure for accurate data retrieval
    try:
        image_bytes = base64.b64decode(response.content[0])  # Example access
        image = Image.open(BytesIO(image_bytes))
        image.save(image_path)
        console.print(f"Image generated and saved as {image_path}")
    except Exception as e:
        console.print(f"Could not process image data: {e}")
