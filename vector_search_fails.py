from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    return np.array(client.embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- Documents with specific codes, emails, numbers ---
documents = [
    "To report security incidents, email security@techcorp.com within 1 hour of discovery.",
    "Error code TX-4021 means the API gateway connection timed out. Restart the service.",
    "Error code TX-4022 indicates an authentication failure. Check your API key.",
    "The monthly revenue for Q3 was $4.2 million, up 15% from Q2.",
    "Machine learning models can be trained using supervised or unsupervised methods.",
    "Contact the HR department at hr-benefits@techcorp.com for benefits questions.",
]

doc_embeddings = [get_embedding(doc) for doc in documents]

# --- Queries where vector search STRUGGLES ---
queries = [
    "What does error TX-4021 mean?",           # exact code lookup
    "email for security team",                  # exact string match
    "hr-benefits@techcorp.com",                 # searching for a literal email
    "How do ML models learn from data?",        # semantic — vector search is GOOD here
]

print("WHERE VECTOR SEARCH STRUGGLES:")
print("=" * 60)

for query in queries:
    query_emb = get_embedding(query)
    scores = [(cosine_similarity(query_emb, de), doc) for de, doc in zip(doc_embeddings, documents)]
    scores.sort(reverse=True)

    print(f"\n🔍 Query: '{query}'")
    print(f"   Top match: [{scores[0][0]:.3f}] {scores[0][1][:70]}...")
    print(f"   2nd match: [{scores[1][0]:.3f}] {scores[1][1][:70]}...")

    # Check if top match is actually correct
    if scores[0][0] < 0.4:
        print(f"   ⚠️  Low confidence! Score is only {scores[0][0]:.3f}")