from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Test 1: Can we talk to OpenAI?
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say 'hello' and nothing else"}],
)
print("LLM says:", response.choices[0].message.content)

# Test 2: Can we create embeddings?
emb = client.embeddings.create(model="text-embedding-3-small", input="hello world")
print("Embedding dimension:", len(emb.data[0].embedding))
print("First 5 values:", emb.data[0].embedding[:5])