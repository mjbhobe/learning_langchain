import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.5,
    messages=[{"role": "user", "content": "Tell me a fun fact"}],
)
print("Fun fact from OpenAI:\n")
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

print("\nFun fact from Google Gemini:\n")
print(gemini_response.choices[0].message.content)
