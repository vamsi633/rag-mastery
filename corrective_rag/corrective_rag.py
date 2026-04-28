from dotenv import load_dotenv
from openai import OpenAI
import os
import json
from pinecone import Pinecone

load_dotenv()
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index=pc.Index(os.getenv("PINECONE_INDEX"))


def retrieve(query,top_k=4):
    emb=client.embeddings.create(model="text-embedding-3-small",input=query).data[0].embedding

    results=index.query(vector=emb,top_k=top_k,include_metadata=True)

    chunks=[]
    for match in results["matches"]:
        chunks.append({
            "text":match["metadata"]["text"],
            "source":match["metadata"].get("source","unknown"),
            "page":match["metadata"].get("page","?"),
            "score":round(match["score"],3),
        })
    return chunks
def grade_chunk(query:str,chunk:dict)->dict:

    content="""You are a relevance grader. Determine if the document chunk
is relevant to answering the user's question.

Grade as:
- "relevant": The chunk contains information that directly helps answer the question
- "irrelevant": The chunk does not help answer the question at all
- "ambiguous": The chunk is tangentially related but may not fully answer the question

Be STRICT. A chunk about "health insurance" is IRRELEVANT to a question about "cryptocurrency".
A chunk mentioning the same topic in passing is AMBIGUOUS, not relevant.

Respond with JSON: {"grade": "relevant/irrelevant/ambiguous", "reason": "brief explanation"}"""

    response=client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":content},
            {"role":"user","content":f"Question: {query}\n\nDocument chunk:\n{chunk['text']}"},
        ],
        temperature=0,
        response_format={"type":"json_object"},
    )

    return json.loads(response.choices[0].message.content)

def grade_all_chunks(query:str,chunks:list[dict])->list[dict]:
    for chunk in chunks:
        grade_result=grade_chunk(query,chunk)
        chunk["grade"]=grade_result.get("grade","ambiguous")
        chunk["grade_reason"]=grade_result.get("reason","")
    return chunks

def decide_action(graded_chunks:list[dict])->dict:
    relevant=[c for c in graded_chunks if c["grade"]=="relevant"]
    ambiguous=[c for c in graded_chunks if c["grade"]=="ambiguous"]
    irrelevant=[c for c in graded_chunks if c["grade"]=="irrelevant"]


    if len(relevant) >= 2:
        return {
            "action": "CORRECT",
            "chunks_to_use": relevant,
            "reason": f"{len(relevant)} relevant chunks found — using standard RAG",
        }
    elif len(relevant) >= 1 or len(ambiguous) >= 2:
        return {
            "action": "AMBIGUOUS",
            "chunks_to_use": relevant + ambiguous,
            "reason": f"{len(relevant)} relevant + {len(ambiguous)} ambiguous — using with caution",
        }
    else:
        return {
            "action": "INCORRECT",
            "chunks_to_use": [],
            "reason": f"No relevant chunks — documents don't contain the answer",
        }
    
def refine_chunks(query:str,chunks:list[dict])->str:
    if not chunks:
        return ""

    all_text = "\n\n".join(
        f"[From {c['source']} page {c['page']}]: {c['text']}" for c in chunks
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Extract ONLY the sentences that are directly relevant to answering the question. Remove everything else. Keep source attributions."
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nDocuments:\n{all_text}"
            },
        ],
        temperature=0,
    )
    return response.choices[0].message.content
def generate(query: str, context: str, action: str) -> str:
    """Generate answer with appropriate confidence level."""

    if action == "INCORRECT":
        return ("I could not find information about this in the available documents. "
                "This topic may not be covered in our current knowledge base.")

    confidence = "high" if action == "CORRECT" else "moderate"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""Answer the question based ONLY on the context.
Confidence level: {confidence}
If confidence is moderate, mention that the information may be incomplete.
Cite sources (document name and page number)."""
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            },
        ],
        temperature=0,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────
# THE COMPLETE CRAG PIPELINE
# ─────────────────────────────────────────────

def corrective_rag(question: str) -> str:
    """
    Full CRAG pipeline:
      1. Retrieve chunks
      2. Grade each chunk
      3. Decide: CORRECT / AMBIGUOUS / INCORRECT
      4. Refine (extract only relevant sentences)
      5. Generate with confidence level
    """
    print(f"\n{'='*60}")
    print(f"❓ {question}")
    print(f"{'='*60}")

    # Step 1: Retrieve
    chunks = retrieve(question)
    print(f"\n  📥 Retrieved {len(chunks)} chunks")

    # Step 2: Grade each chunk
    print(f"\n  📊 Grading chunks...")
    graded = grade_all_chunks(question, chunks)

    for i, chunk in enumerate(graded):
        grade = chunk["grade"].upper()
        reason = chunk["grade_reason"]
        icon = "✅" if grade == "RELEVANT" else "❌" if grade == "IRRELEVANT" else "🟡"
        print(f"    {icon} Chunk {i+1} [{grade}]: {reason}")
        print(f"       Text: {chunk['text'][:60]}...")

    # Step 3: Decide
    decision = decide_action(graded)
    action = decision["action"]
    usable_chunks = decision["chunks_to_use"]

    print(f"\n  🎯 Decision: {action}")
    print(f"     {decision['reason']}")
    print(f"     Using {len(usable_chunks)} of {len(chunks)} chunks")

    # Step 4: Refine
    if usable_chunks:
        print(f"\n  ✂️  Refining chunks...")
        refined_context = refine_chunks(question, usable_chunks)
    else:
        refined_context = ""

    # Step 5: Generate
    print(f"\n  🤖 Generating answer (action: {action})...")
    answer = generate(question, refined_context, action)

    return answer


# ─────────────────────────────────────────────
# TEST — same questions as before
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Questions our docs CAN answer
    print("\n" + "=" * 60)
    print("QUESTIONS OUR DOCS CAN ANSWER:")
    print("=" * 60)

    answer = corrective_rag("What is the remote work policy?")
    print(f"\n  💬 {answer}")

    answer = corrective_rag("What are the Q4 strategic priorities?")
    print(f"\n  💬 {answer}")

    # Questions our docs CANNOT answer
    print("\n" + "=" * 60)
    print("QUESTIONS OUR DOCS CANNOT ANSWER:")
    print("=" * 60)

    answer = corrective_rag("What is NovaTech's policy on cryptocurrency payments?")
    print(f"\n  💬 {answer}")

    answer = corrective_rag("How does NovaTech handle GDPR compliance for EU customers?")
    print(f"\n  💬 {answer}")

    answer = corrective_rag("What programming languages does the engineering team use?")
    print(f"\n  💬 {answer}")

