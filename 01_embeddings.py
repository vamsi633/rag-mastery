from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(response.data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- Let's embed 5 sentences ---
sentences = [
    "I love dogs",
    "I adore puppies",
    "The stock market crashed yesterday",
    "Python is a programming language",
    "I really like canines",
]

embeddings = [get_embedding(s) for s in sentences]

print(f"Each embedding has {len(embeddings[0])} numbers\n")

# --- Compare every pair ---
print("SIMILARITY SCORES:")
print("-" * 50)
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"{sim:.3f}  '{sentences[i]}' vs '{sentences[j]}'")