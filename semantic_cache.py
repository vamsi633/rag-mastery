"""
Project 10: Semantic Cache RAG
"""

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import psycopg2
import time
import json
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))
DATABASE_URL = os.getenv("DATABASE_URL")


class SemanticCache:
    def __init__(self, similarity_threshold=0.7):
        self.threshold = similarity_threshold
        self._setup_table()

    def _setup_table(self):
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS semantic_cache (
                id SERIAL PRIMARY KEY,
                query TEXT NOT NULL,
                query_embedding VECTOR(1536),
                answer TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                hit_count INTEGER DEFAULT 0,
                ttl_hours INTEGER DEFAULT 24
            )
        """)
        conn.commit()
        conn.close()

    def lookup(self, query: str) -> dict | None:
        emb = client.embeddings.create(
            model="text-embedding-3-small", input=query
        ).data[0].embedding
        emb_str = "[" + ",".join(str(x) for x in emb) + "]"

        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT id, query, answer, hit_count,
                   1 - (query_embedding <=> '{emb_str}') AS similarity
            FROM semantic_cache
            WHERE created_at > NOW() - INTERVAL '24 hours'
            ORDER BY query_embedding <=> '{emb_str}'
            LIMIT 1
        """)

        row = cursor.fetchone()

        if row:
            cache_id, cached_query, cached_answer, hit_count, similarity = row
            print(f"     [Cache] Best match: '{cached_query[:50]}...' similarity: {similarity:.4f} (threshold: {self.threshold})")

            if similarity >= self.threshold:
                cursor.execute(
                    "UPDATE semantic_cache SET hit_count = hit_count + 1 WHERE id = %s",
                    (cache_id,)
                )
                conn.commit()
                conn.close()
                return {
                    "answer": cached_answer,
                    "cached_query": cached_query,
                    "similarity": round(similarity, 4),
                    "hit_count": hit_count + 1,
                }

        conn.close()
        return None

    def store(self, query: str, answer: str):
        emb = client.embeddings.create(
            model="text-embedding-3-small", input=query
        ).data[0].embedding
        emb_str = "[" + ",".join(str(x) for x in emb) + "]"

        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO semantic_cache (query, query_embedding, answer)
               VALUES (%s, %s, %s)""",
            (query, emb_str, answer[:3000]),
        )
        conn.commit()
        conn.close()

    def get_stats(self) -> dict:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), COALESCE(SUM(hit_count), 0) FROM semantic_cache")
        total_entries, total_hits = cursor.fetchone()
        conn.close()
        return {"entries": total_entries or 0, "total_hits": int(total_hits or 0)}

    def clear(self):
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM semantic_cache")
        conn.commit()
        conn.close()


def run_rag(question: str) -> str:
    """Standard RAG — runs on cache miss."""
    emb = client.embeddings.create(
        model="text-embedding-3-small", input=question
    ).data[0].embedding

    results = index.query(vector=emb, top_k=4, include_metadata=True)

    context = ""
    for match in results["matches"]:
        text = match["metadata"]["text"]
        source = match["metadata"].get("source", "?")
        page = match["metadata"].get("page", "?")
        context += f"[{source} p{page}] {text}\n\n"

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        sql_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """If this question needs database data, write ONE PostgreSQL query.
Tables: olist_orders_dataset, olist_order_items_dataset, olist_order_payments_dataset, 
olist_order_reviews_dataset, olist_products_dataset, olist_sellers_dataset
Data is from 2016-2018. Don't filter by recent dates. Use double quotes for columns.
If no SQL needed, respond with just: NONE"""
                },
                {"role": "user", "content": question},
            ],
            temperature=0,
        )

        sql = sql_response.choices[0].message.content.strip()
        if sql != "NONE" and sql.upper().startswith("SELECT"):
            sql = sql.replace("```sql", "").replace("```", "").strip()
            cursor.execute("SET statement_timeout = '10000'")
            cursor.execute(sql)
            columns = [d[0] for d in cursor.description]
            rows = cursor.fetchall()
            context += "\nDatabase results:\n"
            for row in rows[:10]:
                context += str(dict(zip(columns, row))) + "\n"

        conn.close()
    except:
        pass

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based on the context. Be concise and cite sources."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


def cached_rag(question: str, cache: SemanticCache) -> dict:
    """Check cache first. If miss, run RAG and store result."""
    start = time.time()

    cached = cache.lookup(question)

    if cached:
        elapsed = round(time.time() - start, 3)
        return {
            "answer": cached["answer"],
            "source": "CACHE HIT",
            "similarity": cached["similarity"],
            "cached_query": cached["cached_query"],
            "time": elapsed,
            "api_calls": 1,
        }

    answer = run_rag(question)
    cache.store(question, answer)

    elapsed = round(time.time() - start, 3)
    return {
        "answer": answer,
        "source": "CACHE MISS",
        "time": elapsed,
        "api_calls": 4,
    }

if __name__ == "__main__":
    cache = SemanticCache(similarity_threshold=0.7)
    cache.clear()

    queries = [
        # Round 1: First time — cache misses
        "What is the total revenue?",
        "What are the top product categories?",
        "What does McKinsey say about AI adoption?",

        # Round 2: Same questions, different words — should be hits
        "What's the overall revenue?",
        "Which product categories make the most money?",
        "According to McKinsey, how widely is AI being adopted?",

        # Round 3: More variations
        "Total sales revenue?",
        "Best selling product categories by revenue?",
    ]

    print("SEMANTIC CACHE RAG DEMO")
    print("=" * 60)

    total_time_cached = 0
    total_time_uncached = 0
    hits = 0
    misses = 0

    for q in queries:
        print(f"\n  Q: {q}")
        result = cached_rag(q, cache)

        source = result["source"]
        elapsed = result["time"]
        api_calls = result["api_calls"]

        if "HIT" in source:
            hits += 1
            total_time_cached += elapsed
            sim = result.get("similarity", 0)
            cached_q = result.get("cached_query", "")
            print(f"  ✅ CACHE HIT ({elapsed}s, {api_calls} API call)")
            print(f"     Matched: '{cached_q[:60]}' (similarity: {sim})")
            print(f"     Answer: {result['answer'][:100]}...")
        else:
            misses += 1
            total_time_uncached += elapsed
            print(f"  ❌ CACHE MISS ({elapsed}s, {api_calls} API calls)")
            print(f"     Answer: {result['answer'][:100]}...")

    stats = cache.get_stats()
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"{'='*60}")
    print(f"  Total queries:    {len(queries)}")
    print(f"  Cache hits:       {hits}")
    print(f"  Cache misses:     {misses}")
    print(f"  Hit rate:         {hits/len(queries)*100:.0f}%")
    print(f"  Cache entries:    {stats['entries']}")

    if misses > 0 and hits > 0:
        avg_miss = total_time_uncached / misses
        avg_hit = total_time_cached / hits
        print(f"\n  Avg MISS time:    {avg_miss:.2f}s")
        print(f"  Avg HIT time:     {avg_hit:.2f}s")
        print(f"  Speedup:          {avg_miss/avg_hit:.1f}x faster on hits")

        cost_without = len(queries) * 0.02
        cost_with = (misses * 0.02) + (hits * 0.0002)
        savings = (1 - cost_with / cost_without) * 100
        print(f"\n  Cost without cache: ${cost_without:.3f}")
        print(f"  Cost with cache:    ${cost_with:.4f}")
        print(f"  Cost savings:       {savings:.0f}%")
    else:
        print(f"\n  No cache hits. Check similarity scores above.")
        print(f"  If scores are below {cache.threshold}, lower the threshold.")