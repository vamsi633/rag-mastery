from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
import chromadb
import re
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─────────────────────────────────────────────
# THE DOCUMENTS
# ─────────────────────────────────────────────

documents = [
    "To report security incidents, email security@techcorp.com within 1 hour of discovery.",
    "Error code TX-4021 means the API gateway connection timed out. Restart the service.",
    "Error code TX-4022 indicates an authentication failure. Check your API key.",
    "The monthly revenue for Q3 was $4.2 million, up 15% from Q2.",
    "Machine learning models can be trained using supervised or unsupervised methods.",
    "Contact the HR department at hr-benefits@techcorp.com for benefits questions.",
    "Supervised learning requires labeled training data to make predictions.",
    "The vacation policy allows 20 days of PTO per year with rollover up to 30 days.",
    "Deep learning uses neural networks with multiple hidden layers for complex tasks.",
    "Passwords must be at least 12 characters and changed every 90 days.",
]

# ─────────────────────────────────────────────
# BUILD BOTH INDEXES
# ─────────────────────────────────────────────

# --- BM25 Index ---
def tokenize(text):
    return re.findall(r'\w+', text.lower())

tokenized_docs = [tokenize(doc) for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# --- ChromaDB Vector Index ---
print("Building ChromaDB vector index...")
chroma = chromadb.Client()
collection = chroma.create_collection("hybrid_docs", metadata={"hnsw:space": "cosine"})

for i, doc in enumerate(documents):
    embedding = client.embeddings.create(
        model="text-embedding-3-small", input=doc
    ).data[0].embedding

    collection.add(
        ids=[f"doc_{i}"],
        embeddings=[embedding],
        documents=[doc],
        metadatas=[{"index": i}],
    )

print(f"Stored {collection.count()} documents in ChromaDB")


# ─────────────────────────────────────────────
# SEARCH FUNCTIONS
# ─────────────────────────────────────────────

def vector_search(query, top_k=5):
    """Semantic search using ChromaDB."""
    query_emb = client.embeddings.create(
        model="text-embedding-3-small", input=query
    ).data[0].embedding

    results = collection.query(query_embeddings=[query_emb], n_results=top_k)

    # Return list of (doc_index, distance)
    ranked = []
    for doc_id, distance in zip(results["ids"][0], results["distances"][0]):
        idx = int(doc_id.split("_")[1])
        ranked.append((idx, distance))
    return ranked


def bm25_search(query, top_k=5):
    """Keyword search using BM25."""
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(i, s) for i, s in ranked[:top_k]]


# ─────────────────────────────────────────────
# RECIPROCAL RANK FUSION
# ─────────────────────────────────────────────

def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
    """
    Combine two ranked lists using RANKS, not scores.

    Formula: RRF_score(doc) = 1/(k + rank) summed across all lists

    WHY THIS WORKS:
    - BM25 scores range 0-5, cosine distances range 0-2
    - You can't add them — the scales are totally different
    - But RANKS are always 1, 2, 3, 4, 5...
    - A document ranked #1 in BOTH systems = very strong signal
    - A document ranked #1 in one, #10 in other = weaker signal
    """
    rrf_scores = {}

    for rank, (doc_idx, _) in enumerate(vector_results):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1.0 / (k + rank + 1)

    for rank, (doc_idx, _) in enumerate(bm25_results):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0) + 1.0 / (k + rank + 1)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked


# ─────────────────────────────────────────────
# HYBRID RAG
# ─────────────────────────────────────────────

def hybrid_rag(question, top_k=3):
    """Complete hybrid search RAG pipeline."""

    vec_results = vector_search(question, top_k=5)
    bm25_results = bm25_search(question, top_k=5)
    fused = reciprocal_rank_fusion(vec_results, bm25_results)

    # Show comparison
    print(f"\n{'='*60}")
    print(f"🔍 Query: {question}")

    print(f"\n  Vector search (ChromaDB) top 3:")
    for rank, (idx, dist) in enumerate(vec_results[:3]):
        print(f"    #{rank+1} [dist:{dist:.3f}] {documents[idx][:55]}...")

    print(f"\n  BM25 search top 3:")
    for rank, (idx, score) in enumerate(bm25_results[:3]):
        print(f"    #{rank+1} [score:{score:.3f}] {documents[idx][:55]}...")

    print(f"\n  HYBRID (RRF) top 3:")
    for rank, (idx, rrf_score) in enumerate(fused[:3]):
        print(f"    #{rank+1} [rrf:{rrf_score:.4f}] {documents[idx][:55]}...")

    # Generate answer with fused results
    top_docs = [documents[idx] for idx, _ in fused[:top_k]]
    context = "\n\n".join(f"[Source {i+1}]: {doc}" for i, doc in enumerate(top_docs))

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based ONLY on the context. Be concise. Cite sources."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


# ─────────────────────────────────────────────
# TEST IT
# ─────────────────────────────────────────────

questions = [
    "What does error TX-4021 mean?",              # exact match — BM25 shines
    "How do AI systems learn from data?",          # semantic — vector shines
    "What is the security email address?",         # mix of both
    "How many vacation days with PTO rollover?",   # mix of both
]

for q in questions:
    answer = hybrid_rag(q)
    print(f"\n💬 Answer: {answer}")