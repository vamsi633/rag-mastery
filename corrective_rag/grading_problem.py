from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import os

load_dotenv()
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index=pc.Index(os.getenv("PINECONE_INDEX"))


def search(query,top_k=3):
    emb=client.embeddings.create(
        model="text-embedding-3-small",input=query
    ).data[0].embedding

    results=index.query(vector=emb,top_k=top_k,include_metadata=True)
    return results["matches"]
def naive_rag(question):
    matches=search(question)

    print(f"\n {question}")
    print(f"\n Retrived chunks:")

    for i,m in enumerate(matches):
        text=m["metadata"]["text"][:80]
        score=round(m["score"],3)
        print(f"[{i+1} (score: {score}) {text}...]")

    context="\n".join(m["metadata"]["text"] for m in matches)

    response=client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"Answer based only on the context.If the context doesn't contain the answer, say so"},
            {"role":"user","content":f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0
    )
    return response.choices[0].message.content


print("=" * 50)
print("QUESTIONS OUR DOCS CAN ANSWER:")
print("=" * 50)

answer = naive_rag("What is the remote work policy?")
print(f"💬 {answer}\n")

answer = naive_rag("What are the Q4 strategic priorities?")
print(f"💬 {answer}\n")

# ── Test with questions our docs CANNOT answer ──
print("=" * 50)
print("QUESTIONS OUR DOCS CANNOT ANSWER:")
print("=" * 50)

answer = naive_rag("What is NovaTech's policy on cryptocurrency payments?")
print(f"💬 {answer}\n")

answer = naive_rag("How does NovaTech handle GDPR compliance for EU customers?")
print(f"💬 {answer}\n")

answer = naive_rag("What programming languages does the engineering team use?")
print(f"💬 {answer}\n")

print("""
WHAT TO NOTICE:
  The retrieval scores are SIMILAR for all questions (0.3-0.5 range)
  Even for questions the docs can't answer, chunks still get retrieved
  because SOME words match ("policy", "team", "engineering")
  
  The LLM sometimes catches it and says "not found"
  But sometimes it hallucinates from vaguely related context
  
  CRAG fixes this by GRADING chunks before the LLM sees them.
""")