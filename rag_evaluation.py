"""
RAG Evaluation — Measure your RAG system quality

Four metrics:
  1. Context Precision — are retrieved docs relevant?
  2. Context Recall — did we find ALL relevant docs?
  3. Faithfulness — is the answer grounded in context?
  4. Answer Relevancy — does the answer address the question?

We'll evaluate our actual RAG system from the unified project.
"""

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import json
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))


# ─────────────────────────────────────────────
# HELPER: Run a basic RAG query and capture everything
# ─────────────────────────────────────────────

def run_rag_with_trace(question: str) -> dict:
    """
    Run RAG and capture every step for evaluation.
    Returns question, retrieved chunks, and generated answer.
    """
    # Retrieve
    emb = client.embeddings.create(
        model="text-embedding-3-small", input=question
    ).data[0].embedding

    results = index.query(vector=emb, top_k=5, include_metadata=True)

    retrieved_chunks = []
    for match in results["matches"]:
        retrieved_chunks.append({
            "text": match["metadata"]["text"],
            "source": match["metadata"].get("source", "?"),
            "page": match["metadata"].get("page", "?"),
            "score": round(match["score"], 3),
        })

    # Generate
    context = "\n\n".join(c["text"] for c in retrieved_chunks)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based ONLY on the context. Be specific and cite facts."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )

    return {
        "question": question,
        "retrieved_chunks": retrieved_chunks,
        "context": context,
        "answer": response.choices[0].message.content,
    }


# ─────────────────────────────────────────────
# METRIC 1: CONTEXT PRECISION
# ─────────────────────────────────────────────

def evaluate_context_precision(question: str, chunks: list[dict]) -> dict:
    """
    For each retrieved chunk, is it ACTUALLY relevant to the question?
    
    Precision = relevant_chunks / total_chunks
    
    High precision (0.8+): Most retrieved docs help answer the question
    Low precision (<0.5): You're retrieving garbage alongside the good stuff
    
    WHY IT MATTERS:
    Low precision → LLM sees irrelevant context → might get confused
    or hallucinate from the irrelevant chunks
    """
    relevance_scores = []

    for i, chunk in enumerate(chunks):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Does this document chunk help answer the question?
Respond with JSON: {"relevant": true/false, "reason": "brief"}
Be strict — tangentially related is NOT relevant."""
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nChunk: {chunk['text'][:500]}"
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        relevance_scores.append({
            "chunk_index": i,
            "relevant": result.get("relevant", False),
            "reason": result.get("reason", ""),
        })

    relevant_count = sum(1 for s in relevance_scores if s["relevant"])
    precision = relevant_count / len(relevance_scores) if relevance_scores else 0

    return {
        "metric": "context_precision",
        "score": round(precision, 2),
        "relevant_chunks": relevant_count,
        "total_chunks": len(relevance_scores),
        "details": relevance_scores,
    }


# ─────────────────────────────────────────────
# METRIC 2: FAITHFULNESS
# ─────────────────────────────────────────────

def evaluate_faithfulness(answer: str, context: str) -> dict:
    """
    Is every claim in the answer supported by the context?
    
    Step 1: Extract individual claims from the answer
    Step 2: Check each claim against the context
    
    Faithfulness = supported_claims / total_claims
    
    WHY IT MATTERS:
    Low faithfulness = the LLM is hallucinating
    It's adding information that isn't in the documents
    """
    # Step 1: Extract claims
    claims_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Extract individual factual claims from this answer.
Each claim should be a single statement that can be verified.
Respond with JSON: {"claims": ["claim 1", "claim 2", ...]}"""
            },
            {"role": "user", "content": answer},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    claims = json.loads(claims_response.choices[0].message.content).get("claims", [])

    if not claims:
        return {"metric": "faithfulness", "score": 1.0, "claims_checked": 0}

    # Step 2: Verify each claim against context
    verified = []
    for claim in claims:
        verify_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Is this claim supported by the context?
Respond with JSON: {"supported": true/false, "reason": "brief"}
The claim must be DIRECTLY supported, not inferred."""
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context[:2000]}\n\nClaim: {claim}"
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(verify_response.choices[0].message.content)
        verified.append({
            "claim": claim,
            "supported": result.get("supported", False),
            "reason": result.get("reason", ""),
        })

    supported_count = sum(1 for v in verified if v["supported"])
    faithfulness = supported_count / len(verified) if verified else 0

    return {
        "metric": "faithfulness",
        "score": round(faithfulness, 2),
        "supported_claims": supported_count,
        "total_claims": len(verified),
        "details": verified,
    }


# ─────────────────────────────────────────────
# METRIC 3: ANSWER RELEVANCY
# ─────────────────────────────────────────────

def evaluate_answer_relevancy(question: str, answer: str) -> dict:
    """
    Does the answer actually address what was asked?
    
    Method: Generate 3 questions that the answer WOULD answer.
    Compare those generated questions with the original question.
    If they're similar → the answer is relevant.
    
    WHY IT MATTERS:
    The answer might be factual and grounded but about the WRONG topic.
    "What is the revenue?" → "NovaTech was founded in 2015" (true but irrelevant)
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Given this answer, rate how well it addresses the question on a scale of 1-10.

10: Directly and completely answers the question
7-9: Mostly answers but might miss some aspects  
4-6: Partially relevant but misses key parts
1-3: Does not address the question at all

Respond with JSON: {"score": 1-10, "reason": "explanation"}"""
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nAnswer: {answer}"
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)

    return {
        "metric": "answer_relevancy",
        "score": round(result.get("score", 0) / 10, 2),
        "reason": result.get("reason", ""),
    }


# ─────────────────────────────────────────────
# FULL EVALUATION
# ─────────────────────────────────────────────

def evaluate_rag(question: str) -> dict:
    """Run RAG and evaluate all metrics."""
    print(f"\n{'='*60}")
    print(f"📝 Evaluating: {question}")
    print(f"{'='*60}")

    # Run RAG
    trace = run_rag_with_trace(question)
    print(f"\n  Answer: {trace['answer'][:150]}...")

    # Metric 1: Context Precision
    print(f"\n  📊 Evaluating Context Precision...")
    precision = evaluate_context_precision(question, trace["retrieved_chunks"])
    print(f"     Score: {precision['score']} ({precision['relevant_chunks']}/{precision['total_chunks']} chunks relevant)")
    for d in precision["details"]:
        icon = "✅" if d["relevant"] else "❌"
        print(f"     {icon} Chunk {d['chunk_index']+1}: {d['reason'][:60]}")

    # Metric 2: Faithfulness
    print(f"\n  📊 Evaluating Faithfulness...")
    faith = evaluate_faithfulness(trace["answer"], trace["context"])
    print(f"     Score: {faith['score']} ({faith.get('supported_claims',0)}/{faith.get('total_claims',0)} claims supported)")
    for d in faith.get("details", []):
        icon = "✅" if d["supported"] else "❌"
        print(f"     {icon} '{d['claim'][:60]}...'")

    # Metric 3: Answer Relevancy
    print(f"\n  📊 Evaluating Answer Relevancy...")
    relevancy = evaluate_answer_relevancy(question, trace["answer"])
    print(f"     Score: {relevancy['score']}")
    print(f"     Reason: {relevancy['reason']}")

    # Overall
    overall = round(
        (precision["score"] + faith["score"] + relevancy["score"]) / 3, 2
    )

    return {
        "question": question,
        "answer": trace["answer"],
        "context_precision": precision["score"],
        "faithfulness": faith["score"],
        "answer_relevancy": relevancy["score"],
        "overall": overall,
    }


# ─────────────────────────────────────────────
# RUN EVALUATION
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Test questions — mix of easy and hard
    test_questions = [
        "What does McKinsey say about generative AI adoption rates?",
        "What are the key cybersecurity risks mentioned in the report?",
        "What is NovaTech's policy on cryptocurrency?",  # not in docs!
    ]

    results = []
    for q in test_questions:
        result = evaluate_rag(q)
        results.append(result)

    # Summary table
    print(f"\n\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Question':<45} {'Precision':>10} {'Faithful':>10} {'Relevant':>10} {'Overall':>10}")
    print(f"{'-'*45} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for r in results:
        q = r["question"][:42] + "..." if len(r["question"]) > 42 else r["question"]
        print(f"{q:<45} {r['context_precision']:>10} {r['faithfulness']:>10} {r['answer_relevancy']:>10} {r['overall']:>10}")

    avg_overall = round(sum(r["overall"] for r in results) / len(results), 2)
    print(f"\n{'Average Overall:':<45} {'':>10} {'':>10} {'':>10} {avg_overall:>10}")

    print(f"""
INTERPRETING SCORES:
  0.8+ → Excellent. Production ready.
  0.6-0.8 → Good but needs improvement.
  0.4-0.6 → Problems. Check chunking, retrieval, prompts.
  <0.4 → Broken. Major issues.

WHAT TO FIX BASED ON LOW SCORES:
  Low Precision → Better chunking, smaller chunks, re-ranking
  Low Faithfulness → Stricter prompts, add "cite sources" instruction  
  Low Relevancy → Better retrieval, query expansion, HyDE
""")