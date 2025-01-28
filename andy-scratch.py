import os
from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-2.0-flash-thinking-exp-01-21",
    contents="high",
    config=types.GenerateContentConfig(
        system_instruction="I say high, you say low",
        temperature=0.1,
    ),
)
print(response.text)
