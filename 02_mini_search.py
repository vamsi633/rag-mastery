from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import os

load_dotenv()
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embeddings(text):
    response=client.embeddings.create(model="text-embedding-3-small",input=text)
    return np.array(response.data[0].embedding)

def cosine_similarity(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))


documents = [
    "Employees can return equipment within 30 days of purchase. Items must be in original packaging.",
    "Our health insurance covers dental and vision for all full-time employees and their families.",
    "The vacation policy allows 20 days of PTO per year. Unused days roll over to the next year.",
    "Performance reviews happen twice a year in June and December. Managers use a 1-5 rating scale.",
    "The office is located at 123 Main Street. Parking is available in the basement garage.",
]

print("embedding 5 documnet chunks...")
doc_embeddings=[get_embeddings(doc) for doc in documents]

query="How many vacation days do i get?"
print(f"\nQuery: '{query}'")


query_embedding=get_embeddings(query)

scores=[]

for i,doc_emb in enumerate(doc_embeddings):
    sim=cosine_similarity(query_embedding,doc_emb)
    scores.append((sim,documents[i]))

scores.sort(reverse=True)

print("RANKED RESULTS:")
print("-"*50)
for rank,(score,doc) in enumerate(scores,1):
    marker="MATCH" if rank==1 else ""
    print(f" #{rank} [{score:.3f}] {doc[:70]}...{marker}")