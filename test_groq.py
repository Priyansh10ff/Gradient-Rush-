import os
import asyncio
from groq import Groq

def test():
    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        print("Client created")
    except Exception as e:
        print("Failed to create client:", e)

test()
