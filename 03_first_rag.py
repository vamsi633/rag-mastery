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

doc_embeddings=[get_embeddings(doc) for doc in documents]

def retrieve(query,top_k=2):
    query_emb=get_embeddings(query)
    scores=[(cosine_similarity(query_emb,doc_emb),doc) for doc_emb,doc in zip(doc_embeddings,documents)]
    scores.sort(reverse=True)
    return [doc for _,doc in scores[:top_k]]

def ask(query):
    print(f"Retriving...")
    retrieved=retrieve(query,top_k=2)
    for i, doc in enumerate(retrieved):
        print(f"Found: {doc[:60]}...")
    context="\n".join(f"-{doc}" for doc in retrieved)
    prompt = f"""Answer the question based ONLY on the context below.
If the context doesn't contain the answer, say "I don't know based on the available information."

Context:
{context}

Question: {query}"""
    
    print(f"Generating answer...")
    response=client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are a helpful HR assistant. Answer concisely"},
            {"role":"user","content":prompt},
        ],
        temperature=0,
    )

    return response.choices[0].message.content

questions=[
    "How many vacation days do I get?",
    "Does insurance cover dental?",
    "What is the meaning of life?",
]

for q in questions:
    answer=ask(q)
    print(f"{answer}")
