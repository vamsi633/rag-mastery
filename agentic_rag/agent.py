"""
The complete Agentic RAG system.
All tools hit PostgreSQL — one database for SQL + vector search + memory.
"""

from dotenv import load_dotenv
from openai import OpenAI
import psycopg2
import json
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DATABASE_URL = os.getenv("DATABASE_URL")


def get_conn():
    return psycopg2.connect(DATABASE_URL)


# ─────────────────────────────────────────────
# TOOL IMPLEMENTATIONS
# ─────────────────────────────────────────────

def get_database_schema() -> str:
    """Discover tables, columns, and sample data."""
    conn = get_conn()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position
    """)
    rows = cursor.fetchall()

    schema = {}
    for table, column, dtype in rows:
        if table not in schema:
            schema[table] = []
        schema[table].append(f"{column} ({dtype})")

    output = "Tables:\n\n"
    for table, columns in schema.items():
        output += f"TABLE: {table}\n  {', '.join(columns)}\n\n"

    # Sample data
    for table in schema:
        if table in ("conversation_memory", "document_chunks"):
            continue
        try:
            cursor.execute(f"SELECT * FROM {table} LIMIT 2")
            cols = [desc[0] for desc in cursor.description]
            for row in cursor.fetchall():
                output += f"Sample {table}: {dict(zip(cols, row))}\n"
        except:
            pass
    
    output += "\nDistinct values for key columns:\n"
    for table in schema:
        if table in ("conversation_memory", "document_chunks"):
            continue
        try:
            cursor.execute(f"""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = '{table}' AND data_type = 'text'
            """)
            text_cols = [r[0] for r in cursor.fetchall()]
            for col in text_cols:
                cursor.execute(f'SELECT DISTINCT "{col}" FROM {table} LIMIT 10')
                values = [r[0] for r in cursor.fetchall() if r[0]]
                if len(values) <= 10:
                    output += f"  {table}.{col}: {values}\n"
        except:
            pass

    conn.close()
    return output


def query_database(sql: str) -> str:
    """Execute a read-only SQL query."""
    sql_upper = sql.upper().strip()
    blocked = ["DROP", "DELETE", "UPDATE", "ALTER", "INSERT", "TRUNCATE"]
    for kw in blocked:
        if kw in sql_upper:
            return f"ERROR: {kw} not allowed. Read-only access."
    if not sql_upper.startswith("SELECT"):
        return "ERROR: Only SELECT queries allowed."

    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("SET statement_timeout = '5000'")
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "No results."

        result = f"Columns: {columns}\n"
        for row in rows[:20]:
            result += str(dict(zip(columns, row))) + "\n"
        if len(rows) > 20:
            result += f"... {len(rows) - 20} more rows"
        return result

    except Exception as e:
        return f"SQL Error: {e}"


def search_documents(query: str) -> str:
    """Vector search over document chunks using pgvector."""
    embedding = client.embeddings.create(
        model="text-embedding-3-small", input=query
    ).data[0].embedding
    emb_str = "[" + ",".join(str(x) for x in embedding) + "]"

    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT text, source, page, embedding <=> '{emb_str}' AS distance
        FROM document_chunks
        ORDER BY embedding <=> '{emb_str}'
        LIMIT 4
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "No documents found."

    output = ""
    for i, (text, source, page, dist) in enumerate(rows):
        relevance = round(1 - dist, 2)
        output += f"[Doc {i+1}, relevance:{relevance}] {text}\n\n"
    return output


def calculate(expression: str) -> str:
    """Safe math calculator."""
    try:
        allowed = set("0123456789.+-*/() ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters"
        return str(round(eval(expression, {"__builtins__": {}}, {}), 2))
    except Exception as e:
        return f"Error: {e}"


def recall_memory(query: str) -> str:
    """Retrieve past conversations."""
    conn = get_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT query, answer, timestamp
        FROM conversation_memory
        ORDER BY timestamp DESC LIMIT 5
    """)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "No previous conversations."

    output = ""
    for q, a, ts in rows:
        output += f"[{ts}] Q: {q}\nA: {a[:200]}...\n\n"
    return output


def save_memory(query, answer, tools_used):
    """Save interaction to memory table."""
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO conversation_memory (query, answer, agent_used, tools_called)
               VALUES (%s, %s, %s, %s)""",
            (query, answer[:2000], "agent", json.dumps(tools_used)),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"  Memory save error: {e}")


# ─────────────────────────────────────────────
# TOOL REGISTRY (dynamic dispatch)
# ─────────────────────────────────────────────

TOOL_FUNCTIONS = {
    "get_database_schema": lambda **kwargs: get_database_schema(),
    "query_database": lambda sql, **kwargs: query_database(sql),
    "search_documents": lambda query, **kwargs: search_documents(query),
    "calculate": lambda expression, **kwargs: calculate(expression),
    "recall_memory": lambda query, **kwargs: recall_memory(query),
}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Semantic search over company documents (Q3 report, policies). Use for strategy, plans, analysis, qualitative info. Search ONE topic at a time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Focused search query about one topic"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_database_schema",
            "description": "Discover tables, columns, types, and sample data. ALWAYS call this BEFORE writing SQL.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "Run a SELECT query on PostgreSQL. For employee, sales, ticket data. Call get_database_schema first. Use double quotes around column names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string", "description": 'PostgreSQL SELECT query'}
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Math calculations. Percentages, totals, comparisons. Don't do mental math.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math like '185000 * 0.15'"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "Search past conversations. Use when user says 'earlier', 'before', 'what did you find'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for in past conversations"}
                },
                "required": ["query"],
            },
        },
    },
]


# ─────────────────────────────────────────────
# THE AGENT LOOP
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI Business Operations Assistant for NovaTech Inc.

You have access to:
1. Company documents (Q3 report) via search_documents — for strategy, plans, qualitative info
2. PostgreSQL database (employees, sales, tickets) via query_database — for numbers, counts, exact data
3. Calculator for math
4. Memory of past conversations

RULES:
- ALWAYS call get_database_schema BEFORE writing any SQL
- Search documents for ONE topic at a time
- For complex questions, gather from MULTIPLE sources before answering
- Use calculate for math — don't do mental math
- Show your reasoning
- Cite sources"""


def run_agent(question, max_steps=8):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    tools_used = []

    print(f"\n{'='*60}")
    print(f"❓ {question}")
    print(f"{'='*60}")

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_DEFINITIONS,
            temperature=0,
        )

        message = response.choices[0].message

        if message.tool_calls:
            messages.append(message)

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                tools_used.append(func_name)

                print(f"\n  Step {step+1} → 🔧 {func_name}")
                for k, v in args.items():
                    print(f"           {k}: {v}")

                # Dynamic dispatch
                func = TOOL_FUNCTIONS.get(func_name)
                result = func(**args) if func else f"Unknown tool: {func_name}"

                preview = result[:200].replace("\n", " ")
                print(f"           → {preview}...")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })
        else:
            answer = message.content
            print(f"\n  ✅ Done in {step+1} steps | Tools: {tools_used}")
            save_memory(question, answer, tools_used)
            return answer

    return "Max steps reached."


# ─────────────────────────────────────────────
# INTERACTIVE MODE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("🤖 NovaTech AI Operations Assistant")
    print("   All data in PostgreSQL (SQL + pgvector)")
    print("   Type 'quit' to exit\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        answer = run_agent(question)
        print(f"\n💬 {answer}\n")