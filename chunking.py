from langchain_text_splitters import RecursiveCharacterTextSplitter

handbook_page = """Company Return Policy

Our return policy allows employees to return company-issued equipment within 30 days of receiving it. All items must be in their original packaging and in working condition. Electronics such as laptops and monitors have a shorter 15-day return window due to rapid depreciation.

To initiate a return, employees should email equipment@company.com with their employee ID and a description of the item. The IT department will respond within 2 business days with a return shipping label.

Refunds are processed within 5-7 business days after the item is received and inspected. If the item shows signs of damage beyond normal wear, a partial refund may be issued at management's discretion.

Employee Health Benefits

All full-time employees are eligible for health insurance starting on their first day. The company offers three plan tiers: Basic, Standard, and Premium. Basic covers medical only. Standard adds dental and vision. Premium includes all of the above plus mental health coverage and gym membership reimbursement.

Dependents can be added during open enrollment in November or within 30 days of a qualifying life event such as marriage or the birth of a child. The company covers 80% of the premium for the employee and 50% for dependents.

For questions about benefits, contact hr-benefits@company.com or visit the Benefits Portal on the company intranet."""


# --- RecursiveCharacterTextSplitter ---
# HOW IT WORKS:
# It tries to split on these separators IN ORDER:
#   1. "\n\n" (paragraph breaks) — best, keeps topics together
#   2. "\n"   (single newlines) — ok, keeps related lines together
#   3. ". "   (sentences) — decent, at least complete sentences
#   4. " "    (words) — last resort, never cuts mid-word
#
# If a chunk is still too big after using a separator,
# it moves to the NEXT separator for that chunk.
# That's why it's called "recursive."

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,       # max characters per chunk
    chunk_overlap=50,     # repeat 50 chars between chunks
    separators=["\n\n", "\n", ". ", " "],
)

chunks = splitter.split_text(handbook_page)

print("RECURSIVE CHUNKING (LangChain):")
print("=" * 50)
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1} ({len(chunk)} chars):")
    print(f"  '{chunk}'")

# --- Now let's see what OVERLAP does ---
print(f"\n\n{'='*50}")
print("WHY OVERLAP MATTERS:")
print("=" * 50)

# Show overlap between chunk 1 and chunk 2
if len(chunks) >= 2:
    end_of_chunk1 = chunks[0][-60:]
    start_of_chunk2 = chunks[1][:60]
    print(f"\n  End of Chunk 1:   '...{end_of_chunk1}'")
    print(f"  Start of Chunk 2: '{start_of_chunk2}...'")
    print(f"\n  See the repeated text? That's the overlap.")
    print(f"  It ensures no information is lost at the boundary.")

# --- Compare different chunk sizes ---
print(f"\n\n{'='*50}")
print("EXPERIMENT: chunk_size effect")
print("=" * 50)
for size in [150, 300, 500, 800]:
    s = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=50)
    c = s.split_text(handbook_page)
    print(f"  chunk_size={size:>4} → {len(c)} chunks (avg {sum(len(x) for x in c)//len(c)} chars each)")

print(f"""
CHOOSING CHUNK SIZE (interview answer):
  - 100-200: Very precise retrieval, but chunks lack context
  - 300-500: Good balance (most common in production)
  - 500-1000: Rich context, but retrieval gets noisy
  - 1000+: Usually too big — multiple topics per chunk

Rule of thumb: Start with 500, chunk_overlap=50.
Adjust based on your retrieval quality metrics.
""")