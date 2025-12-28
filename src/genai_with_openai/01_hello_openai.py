import os
from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
console = Console()

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = openai.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.5,
    messages=[{"role": "user", "content": "Tell me a fun fact"}],
)
console.print("[sky_blue1]Fun fact from OpenAI:[/sky_blue1]\n")
print(response.choices[0].message.content)

# I can use a Google Gemini model with OpenAI API as well
gemini = OpenAI(
    base_url=os.getenv("GEMINI_BASE_URL"),
    api_key=os.getenv("GOOGLE_API_KEY"),
)

gemini_response = gemini.chat.completions.create(
    model="gemini-2.5-flash",
    temperature=0.5,
    messages=[{"role": "user", "content": "Tell me a fun fact"}],
)

console.print("[sky_blue1]\nFun fact from Google Gemini:[/sky_blue1]\n")
console.print(Markdown(gemini_response.choices[0].message.content))

# here is a more practical example with OpenAI
system_message = """
You are an AI assistant that helps humans by generating tutorials given a text.
You will be provided with a text. If the text contains any kind of
istructions on how to proceed with something, generate a tutorial in a
bullet list with markdown formatting.
Otherwise, inform the user that the text does not contain any instructions.
Text:
"""

user_message = """
To prepare the known sauce from Genova, Italy, you can start by toasting
the pine nuts to then coarsely chop them in a kitchen mortar together with 
basil and garlic. Then, add half of the oil in the kitchen mortar and 
season with salt and pepper.
Finally, transfer the pesto to a bowl and stir in the grated Parmesan
cheese.
"""

response = openai.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.5,
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ],
)

# response = gemini.chat.completions.create(
#     model="gemini-2.5-flash",
#     temperature=0.5,
#     messages=[
#         {"role": "system", "content": system_message},
#         {"role": "user", "content": user_message},
#     ],
# )

console.print("\n[sky_blue1]Tutorial from OpenAI:[/sky_blue1]\n")
console.print(Markdown(response.choices[0].message.content))

response = openai.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.5,
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": "The sun is shining and the birds are chirping"},
    ],
)

console.print("\n[sky_blue1]Tutorial #2 from OpenAI:[/sky_blue1]\n")
console.print(Markdown(response.choices[0].message.content))