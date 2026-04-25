# First install: pip install rank-bm25

from rank_bm25 import BM25Okapi
import re

documents = [
    "To report security incidents, email security@techcorp.com within 1 hour of discovery.",
    "Error code TX-4021 means the API gateway connection timed out. Restart the service.",
    "Error code TX-4022 indicates an authentication failure. Check your API key.",
    "The monthly revenue for Q3 was $4.2 million, up 15% from Q2.",
    "Machine learning models can be trained using supervised or unsupervised methods.",
    "Contact the HR department at hr-benefits@techcorp.com for benefits questions.",
]

# --- BM25: How it works ---
# 1. Tokenize each document into words
# 2. For each query word, check:
#    - How often does it appear in THIS document? (term frequency)
#    - How RARE is it across ALL documents? (inverse document frequency)
# 3. Rare words that appear in a document = strong match
#
# "TX-4021" appears in only ONE document → very rare → strong signal
# "the" appears everywhere → not useful → low signal

def tokenize(text):
    """Simple tokenizer: lowercase, split on non-alphanumeric."""
    return re.findall(r'\w+', text.lower())

# Tokenize all documents
tokenized_docs = [tokenize(doc) for doc in documents]

# Build BM25 index
bm25 = BM25Okapi(tokenized_docs)

# --- Same queries that vector search struggled with ---
queries = [
    "What does error TX-4021 mean?",
    "email for security team",
    "hr-benefits@techcorp.com",
    "How do ML models learn from data?",
]

print("BM25 KEYWORD SEARCH:")
print("=" * 60)

for query in queries:
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Rank documents by score
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    print(f"\n🔍 Query: '{query}'")
    print(f"   Tokens: {tokenized_query}")
    for rank, (idx, score) in enumerate(ranked[:3]):
        marker = "✅" if rank == 0 and score > 0 else "  "
        print(f"   {marker} [{score:.3f}] {documents[idx][:70]}...")

print(f"""

COMPARE WITH VECTOR SEARCH:
{'='*60}
Query: "What does error TX-4021 mean?"
  Vector search: ❌ Returned TX-4022 (wrong code!)
  BM25:          ✅ Returns TX-4021 (exact match on the code)

Query: "How do ML models learn from data?"  
  Vector search: ✅ Great semantic match
  BM25:          🤷 Decent but relies on word overlap

CONCLUSION:
  Vector search = understands MEANING (good for concepts)
  BM25 = matches exact WORDS (good for codes, names, emails)
  
  HYBRID = use BOTH and combine the results!
""")