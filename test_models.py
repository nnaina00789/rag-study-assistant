import os, time
from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

candidates = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
              "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite",
              "gemini-2.5-flash-lite", "gemini-flash-latest", "gemini-flash-lite-latest"]

for name in candidates:
    try:
        r = client.models.generate_content(model=name, contents="Say hello in one word.")
        print(f"OK    {name}: {r.text.strip()}")
    except errors.APIError as e:
        print(f"FAIL  {name}: {e.code}")
    time.sleep(1)