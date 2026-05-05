"""
Project 8: Adaptive RAG

Classifies query complexity → routes to the right STRATEGY.

SIMPLE   → Direct answer (LLM knowledge) or single retrieval
MODERATE → Standard RAG with one agent
COMPLEX  → Multi-agent + CRAG + synthesis

This is different from Project 6's router:
  Router picks WHICH agent (sales vs HR vs research)
  Adaptive picks WHICH STRATEGY (cheap vs medium vs expensive)
"""

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
import psycopg2
import json
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))
DATABASE_URL = os.getenv("DATABASE_URL")


# ─────────────────────────────────────────────
# STEP 1: CLASSIFY query complexity
# ─────────────────────────────────────────────

def classify_query(question: str) -> dict:
    """
    Classify a question into SIMPLE / MODERATE / COMPLEX.

    SIMPLE:
      - General knowledge questions
      - Definitions, explanations
      - Single fact lookups
      Examples: "What is generative AI?", "Define machine learning"

    MODERATE:
      - Needs ONE data source (SQL or document search)
      - Single-step retrieval
      - Specific data questions
      Examples: "Top 5 products by revenue", "What does the report say about AI?"

    COMPLEX:
      - Needs MULTIPLE data sources
      - Multi-step analysis
      - Comparison, synthesis, report generation
      - Contains words like "analyze", "compare", "plan", "report"
      Examples: "Compare sales trends with customer satisfaction and suggest improvements"
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Classify the query complexity. Consider:

SIMPLE: General knowledge, definitions, single facts the LLM already knows.
  No database or document search needed.
  
MODERATE: Needs ONE data lookup — either a database query OR a document search.
  Single-step retrieval. One specific question.

COMPLEX: Needs MULTIPLE data sources, multi-step analysis, comparison, 
  report generation, or combining database data with document insights.

Respond with JSON:
{
    "complexity": "simple/moderate/complex",
    "reasoning": "brief explanation",
    "needs_database": true/false,
    "needs_documents": true/false,
    "estimated_steps": 1-10
}"""
            },
            {"role": "user", "content": question},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ─────────────────────────────────────────────
# STRATEGY 1: DIRECT (no retrieval)
# ─────────────────────────────────────────────

def strategy_direct(question: str) -> str:
    """
    Just ask the LLM. No tools, no retrieval, no cost.
    Used for general knowledge and definitions.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer concisely and accurately."},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────
# STRATEGY 2: SINGLE RETRIEVAL (one search + answer)
# ─────────────────────────────────────────────

def strategy_single_retrieval(question: str, source: str) -> str:
    """
    One retrieval step, one generation step.
    No agent loop. No tool calling. Minimal cost.

    source: "database" or "documents"
    """
    if source == "documents":
        # Search Pinecone
        emb = client.embeddings.create(
            model="text-embedding-3-small", input=question
        ).data[0].embedding

        results = index.query(vector=emb, top_k=4, include_metadata=True)

        context = ""
        for match in results["matches"]:
            text = match["metadata"]["text"]
            source_file = match["metadata"].get("source", "?")
            page = match["metadata"].get("page", "?")
            context += f"[{source_file} p{page}] {text}\n\n"

    elif source == "database":
        # Let LLM write ONE SQL query
        # First get schema
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
            AND table_name LIKE 'olist%'
            ORDER BY table_name, ordinal_position
        """)
        schema_rows = cursor.fetchall()
        conn.close()

        schema = {}
        for table, col, dtype in schema_rows:
            if table not in schema:
                schema[table] = []
            schema[table].append(f"{col}({dtype})")

        schema_str = "\n".join(f"{t}: {', '.join(c)}" for t, c in schema.items())

        # Generate SQL
        sql_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""Write ONE PostgreSQL SELECT query to answer the question.
Schema:
{schema_str}

NOTE: Data is from 2016-2018. Don't filter by recent dates.
Use double quotes for column names. Respond with ONLY the SQL, nothing else."""
                },
                {"role": "user", "content": question},
            ],
            temperature=0,
        )

        sql = sql_response.choices[0].message.content.strip()
        sql = sql.replace("```sql", "").replace("```", "").strip()

        # Execute
        try:
            conn = psycopg2.connect(DATABASE_URL)
            cursor = conn.cursor()
            cursor.execute("SET statement_timeout = '10000'")
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            conn.close()

            context = f"SQL: {sql}\n\nResults:\n"
            for row in rows[:15]:
                context += str(dict(zip(columns, row))) + "\n"
        except Exception as e:
            context = f"SQL failed: {e}\nQuery was: {sql}"

    else:
        context = "No source specified."

    # Generate answer from context
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based on the context. Be concise. Cite sources if available."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────
# STRATEGY 3: FULL PIPELINE (multi-agent + CRAG)
# ─────────────────────────────────────────────

def strategy_full_pipeline(question: str) -> str:
    """
    Full multi-agent pipeline with CRAG and synthesis.
    Imports from the unified project.
    Most expensive but highest quality.
    """
    # Import from unified project
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "autonomous_corrective_multiagent_rag"))

    try:
        from core.router import route
        from core.synthesizer import synthesize
        from agents.registry import run_agent, AGENTS

        routing = route(question)
        agents = routing.get("agents", ["research"])
        sub_questions = routing.get("sub_questions", {})

        agent_results = {}
        for agent_name in agents:
            if agent_name not in AGENTS:
                continue
            sub_q = sub_questions.get(agent_name, question)
            result = run_agent(agent_name, sub_q)
            agent_results[agent_name] = result

        if len(agent_results) == 1:
            return list(agent_results.values())[0]
        else:
            return synthesize(question, agent_results)

    except ImportError:
        # Fallback if unified project not available
        return strategy_single_retrieval(question, "documents")


# ─────────────────────────────────────────────
# THE ADAPTIVE ROUTER
# ─────────────────────────────────────────────

def adaptive_rag(question: str) -> str:
    """
    The complete Adaptive RAG pipeline:
    1. Classify query complexity
    2. Pick the right strategy
    3. Execute with minimum cost
    """
    print(f"\n{'='*60}")
    print(f"❓ {question}")
    print(f"{'='*60}")

    # Step 1: Classify
    classification = classify_query(question)
    complexity = classification.get("complexity", "moderate")
    reasoning = classification.get("reasoning", "")
    needs_db = classification.get("needs_database", False)
    needs_docs = classification.get("needs_documents", False)
    est_steps = classification.get("estimated_steps", 3)

    print(f"\n  🏷️  Complexity: {complexity.upper()}")
    print(f"     Reason: {reasoning}")
    print(f"     Needs DB: {needs_db} | Needs Docs: {needs_docs}")
    print(f"     Estimated steps: {est_steps}")

    # Step 2: Pick strategy
    if complexity == "simple":
        print(f"\n  ⚡ Strategy: DIRECT (no retrieval)")
        answer = strategy_direct(question)
        strategy_used = "direct"

    elif complexity == "moderate":
        if needs_docs and not needs_db:
            print(f"\n  📄 Strategy: SINGLE RETRIEVAL (documents)")
            answer = strategy_single_retrieval(question, "documents")
            strategy_used = "single_retrieval_docs"
        elif needs_db and not needs_docs:
            print(f"\n  🗄️  Strategy: SINGLE RETRIEVAL (database)")
            answer = strategy_single_retrieval(question, "database")
            strategy_used = "single_retrieval_db"
        else:
            print(f"\n  🗄️  Strategy: SINGLE RETRIEVAL (database)")
            answer = strategy_single_retrieval(question, "database")
            strategy_used = "single_retrieval_db"

    else:  # complex
        print(f"\n  🔥 Strategy: FULL PIPELINE (multi-agent + CRAG)")
        answer = strategy_full_pipeline(question)
        strategy_used = "full_pipeline"

    print(f"\n  ✅ Strategy used: {strategy_used}")
    return answer


# ─────────────────────────────────────────────
# TEST IT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    queries = [
        # SIMPLE — should use direct (no retrieval)
        "What is machine learning?",

        # MODERATE — should use single retrieval (database)
        "What are the top 5 product categories by revenue?",

        # MODERATE — should use single retrieval (documents)
        "What does the McKinsey report say about generative AI adoption?",

        # COMPLEX — should use full pipeline
        "Analyze our e-commerce revenue trends by region and compare with McKinsey's recommendations for digital transformation",
    ]

    for q in queries:
        answer = adaptive_rag(q)
        print(f"\n💬 {answer[:300]}...")
        print(f"\n{'─'*60}")