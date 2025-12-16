from dotenv import load_dotenv
import google.genai as genai

load_dotenv()

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.5-pro",
    contents="Explain how AI works in a few words to a newbie"
)

print(response.content)
