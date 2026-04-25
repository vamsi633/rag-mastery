from dotenv import load_dotenv
import os
from openai import OpenAI
import chromadb

load_dotenv()
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

chroma_client=chromadb.Client()

collection=chroma_client.create_collection(
    name="company_handbook",
    metadata={"hnsw:space":"cosine"},
)
documents = [
    "Employees can return equipment within 30 days of purchase. Items must be in original packaging.",
    "Our health insurance covers dental and vision for all full-time employees and their families.",
    "The vacation policy allows 20 days of PTO per year. Unused days roll over to the next year.",
    "Performance reviews happen twice a year in June and December. Managers use a 1-5 rating scale.",
    "The office is located at 123 Main Street. Parking is available in the basement garage.",
    "Remote work is allowed 3 days per week. Employees must be in office on Tuesday and Thursday.",
    "The company matches 401k contributions up to 5 percent of salary.",
    "New employees get a laptop and monitor on their first day. Equipment must be returned upon leaving.",
    "Sick leave provides 10 days per year. A doctor's note is required for absences longer than 3 days.",
    "The annual company retreat is held in September. All employees are expected to attend.",
]

print("Embeddings and storing 10 documents...")

for i, doc in enumerate(documents):
    embeddings=client.embeddings.create(
        model="text-embedding-3-small",
        input=doc
    ).data[0].embedding

    collection.add(
        ids=[f"doc_{i}"],
        embeddings=embeddings,
        documents=[doc],
        metadatas=[{"source":"handbook","chunk_index":i}],
    )
print(f"stored {collection.count()} documents in chromadb\n")

def ask(query,top_k=2):
    query_embedding=client.embeddings.create(model="text-embedding-3-small",input=query).data[0].embedding

    results=collection.query(query_embeddings=[query_embedding],n_results=top_k)

    retrived_docs=results["documents"][0]
    distances=results["distances"][0]

    print(f" query: {query}")
    print(f"Retrieved {len(retrived_docs)} chunks:")
    for doc,dist in zip(retrived_docs,distances):
        print(f" [{dist:.3f}] {doc[:70]}...")
    
    context="\n".join(f"- {doc}" for doc in retrived_docs)
    prompt = f"""Answer the question based ONLY on the context below.
If the context doesn't contain the answer, say "I don't know."

Context:
{context}

Question: {query}"""
    
    response=client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are a helpful HR assistant.But concise"},
            {"role":"user","content":prompt},
        ],
        temperature=0,
    )
    return response.choices[0].message.content

questions=[
    "Can I work from home?",
    "What happens to my unused vacation days?",
    "Does the company help with retirement savings?",
    "How do I get a parking spot?",
]

for q in questions:
    answer=ask(q)
    print(f" {answer}")
    

    