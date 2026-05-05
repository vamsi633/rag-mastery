"""
Project 8 — Step 1: Why Adaptive RAG matters

Same question, three different strategies.
Shows the cost/quality tradeoff.
"""

from dotenv import load_dotenv
from openai import OpenAI
import time
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def count_tokens(messages):
    """Rough token estimate."""
    return sum(len(m.get("content", "")) // 4 for m in messages if isinstance(m.get("content"), str))


# ── Strategy 1: DIRECT (no retrieval, no tools) ──

def strategy_direct(question):
    """Just ask the LLM. No tools, no retrieval."""
    start = time.time()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer concisely."},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )

    elapsed = round(time.time() - start, 2)
    return {
        "answer": response.choices[0].message.content[:200],
        "time": elapsed,
        "api_calls": 1,
        "strategy": "DIRECT",
    }


# ── Strategy 2: SINGLE RETRIEVAL (one search + answer) ──

def strategy_single(question):
    """One retrieval, one generation. No agent loop."""
    start = time.time()
    api_calls = 0

    # Simulate: embed query + search (1 call)
    api_calls += 1
    context = "[Simulated search result about e-commerce revenue trends]"

    # Generate answer (1 call)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer using ONLY the context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    api_calls += 1

    elapsed = round(time.time() - start, 2)
    return {
        "answer": response.choices[0].message.content[:200],
        "time": elapsed,
        "api_calls": api_calls,
        "strategy": "SINGLE_RETRIEVAL",
    }


# ── Strategy 3: FULL PIPELINE (router + agents + CRAG + synthesis) ──

def strategy_full(question):
    """Multi-agent with grading and synthesis."""
    start = time.time()
    api_calls = 0

    # Router (1 call)
    api_calls += 1

    # Agent 1: get_schema + query (2 calls)
    api_calls += 2

    # Agent 2: search + grade 4 chunks (5 calls)
    api_calls += 5

    # Synthesize (1 call)
    api_calls += 1

    # Simulate final answer
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": f"Simulate answering: {question}"},
        ],
        temperature=0,
    )
    api_calls += 1

    elapsed = round(time.time() - start, 2)
    return {
        "answer": response.choices[0].message.content[:200],
        "time": elapsed,
        "api_calls": api_calls,
        "strategy": "FULL_PIPELINE",
    }


# ── Compare strategies for different query types ──

queries = {
    "SIMPLE": "What is generative AI?",
    "MODERATE": "What are the top 5 product categories by revenue?",
    "COMPLEX": "Analyze our business performance and create an improvement plan using McKinsey technology trends",
}

print("STRATEGY COMPARISON:")
print("=" * 70)
print(f"{'Query Type':<12} {'Strategy':<20} {'API Calls':<12} {'Time':<8} {'Cost Est.'}")
print("-" * 70)

for complexity, question in queries.items():
    results = [
        strategy_direct(question),
        strategy_single(question),
        strategy_full(question),
    ]

    for r in results:
        cost = r["api_calls"] * 0.002  # rough estimate per call
        marker = " ← BEST" if (
            (complexity == "SIMPLE" and r["strategy"] == "DIRECT") or
            (complexity == "MODERATE" and r["strategy"] == "SINGLE_RETRIEVAL") or
            (complexity == "COMPLEX" and r["strategy"] == "FULL_PIPELINE")
        ) else ""
        print(f"{complexity:<12} {r['strategy']:<20} {r['api_calls']:<12} {r['time']:<8} ${cost:.3f}{marker}")

    print()

print("""
THE POINT:
  SIMPLE queries don't need 10 API calls → use DIRECT
  MODERATE queries don't need multi-agent → use SINGLE RETRIEVAL
  COMPLEX queries need the full pipeline → use FULL PIPELINE

  Without Adaptive RAG:
    Every query uses FULL PIPELINE → expensive, slow
    
  With Adaptive RAG:
    SIMPLE (60% of traffic) → 1 API call
    MODERATE (30%)          → 2-3 API calls
    COMPLEX (10%)           → 10+ API calls
    
    You save ~70% on API costs by not over-processing simple queries.
""")