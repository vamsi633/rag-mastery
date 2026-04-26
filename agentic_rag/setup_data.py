"""
Sets up ALL data in PostgreSQL:
  - Structured data from CSVs (employees, sales, tickets)
  - Document chunks with embeddings from PDF (pgvector)
  
One database. No ChromaDB needed.
"""

from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import psycopg2
import fitz
import csv
import os
import random
from datetime import datetime, timedelta

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DATABASE_URL = os.getenv("DATABASE_URL")


# ─────────────────────────────────────────────
# STEP 1: Generate CSV data
# ─────────────────────────────────────────────

def generate_csvs():
    """Generate realistic business data CSVs."""
    os.makedirs("data", exist_ok=True)

    # ── Employees ──
    departments = {
        "Engineering": ["Junior Engineer", "Engineer", "Senior Engineer", "Staff Engineer", "Engineering Manager"],
        "Sales": ["Sales Rep", "Senior Sales Rep", "Account Executive", "Sales Manager"],
        "Customer Success": ["CS Representative", "CS Manager"],
        "Product": ["Product Manager", "Senior PM"],
        "Marketing": ["Marketing Specialist", "Content Manager"],
        "HR": ["HR Coordinator", "HR Manager"],
    }

    salary_map = {
        "Junior": (70000, 95000), "Rep": (70000, 100000), "Specialist": (75000, 100000),
        "Coordinator": (60000, 80000), "Engineer": (120000, 150000), "Senior": (145000, 180000),
        "Staff": (170000, 200000), "Manager": (130000, 175000), "Account": (100000, 140000),
        "Content": (85000, 110000),
    }

    first_names = ["Sarah", "Marcus", "Priya", "Tom", "Yuki", "Alex", "Emma", "James", "Nina", "David",
                   "Lisa", "Robert", "Jennifer", "Carlos", "Amy", "Brian", "Diana", "Mike", "Rachel", "Sam",
                   "Olivia", "Kevin", "Hannah", "Chris", "Maria", "John", "Sophie", "Daniel", "Laura", "Ryan"]
    last_names = ["Chen", "Johnson", "Patel", "Wilson", "Tanaka", "Rivera", "Liu", "Brown", "Kowalski", "Park",
                  "Wang", "Kim", "Adams", "Mendez", "Zhang", "Foster", "Ross", "Murphy", "Green", "Martin"]
    offices = ["San Francisco", "New York", "London", "Tokyo", "Bangalore"]

    employees = []
    emp_id = 1
    used_names = set()

    for dept, roles in departments.items():
        count = random.randint(6, 12) if dept == "Engineering" else random.randint(3, 6)
        for _ in range(count):
            while True:
                name = f"{random.choice(first_names)} {random.choice(last_names)}"
                if name not in used_names:
                    used_names.add(name)
                    break
            role = random.choice(roles)
            salary_key = next((k for k in salary_map if k in role), "Engineer")
            low, high = salary_map[salary_key]
            salary = round(random.uniform(low, high), -3)
            days_ago = random.randint(30, 1500)
            hire_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            status = random.choices(["active", "left"], weights=[85, 15])[0]
            office = random.choice(offices)
            employees.append([emp_id, name, dept, role, salary, hire_date, status, office])
            emp_id += 1

    with open("data/employees.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "department", "role", "salary", "hire_date", "status", "office"])
        w.writerows(employees)

    # ── Sales ──
    customers = [
        ("Acme Corp", "North America"), ("BMW Group", "Europe"), ("Sony Corp", "Asia Pacific"),
        ("JPMorgan Chase", "North America"), ("Petrobras", "Latin America"),
        ("Samsung Electronics", "Asia Pacific"), ("Barclays PLC", "Europe"),
        ("Toyota Motor", "Asia Pacific"), ("Walmart Inc", "North America"),
        ("Banco do Brasil", "Latin America"), ("Siemens AG", "Europe"),
        ("Netflix Inc", "North America"), ("Tesla Inc", "North America"),
        ("Infosys", "Asia Pacific"), ("Spotify", "Europe"),
    ]
    products = ["CloudSync Enterprise", "DevTools Pro", "NovaTech Analytics", "Mobile SDK"]
    statuses = ["closed_won", "closed_won", "closed_won", "negotiating", "proposal", "lost"]
    sales_reps = [e[1] for e in employees if e[2] == "Sales"]
    if not sales_reps:
        sales_reps = ["Direct Sale"]

    sales = []
    for i, (customer, region) in enumerate(customers):
        product = random.choice(products)
        amount = round(random.uniform(25000, 300000), -3)
        status = random.choice(statuses)
        close_date = (datetime.now() - timedelta(days=random.randint(1, 90))).strftime("%Y-%m-%d") if "closed" in status else ""
        rep = random.choice(sales_reps)
        sales.append([i + 1, f"{customer} - {product}", customer, amount, region, rep, status, close_date, product])

    with open("data/sales.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "deal_name", "customer", "amount", "region", "sales_rep", "status", "close_date", "product"])
        w.writerows(sales)

    # ── Tickets ──
    issues = [
        ("Dashboard not loading after update", "high"),
        ("Data sync delay between regions", "critical"),
        ("API rate limiting errors", "high"),
        ("SSO integration failing", "critical"),
        ("Language support missing", "medium"),
        ("Bulk export timing out", "high"),
        ("Compliance report failing", "critical"),
        ("Documentation unclear", "low"),
        ("Custom widgets not saving", "medium"),
        ("Feature request: collaboration", "low"),
        ("Login timeout on mobile", "high"),
        ("Webhook notifications delayed", "medium"),
    ]
    cs_agents = [e[1] for e in employees if e[2] == "Customer Success"]
    if not cs_agents:
        cs_agents = ["Unassigned"]
    ticket_customers = [c[0] for c in customers[:8]]

    tickets = []
    for i, (issue, priority) in enumerate(issues):
        customer = random.choice(ticket_customers)
        product = random.choice(products)
        status = random.choices(["resolved", "in_progress", "open"], weights=[50, 30, 20])[0]
        assigned = random.choice(cs_agents) if status != "open" else ""
        created = (datetime.now() - timedelta(days=random.randint(1, 45))).strftime("%Y-%m-%d")
        resolved = (datetime.now() - timedelta(days=random.randint(0, 3))).strftime("%Y-%m-%d") if status == "resolved" else ""
        tickets.append([i + 1, customer, issue, product, priority, status, assigned, created, resolved])

    with open("data/tickets.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "customer", "issue", "product", "priority", "status", "assigned_to", "created_date", "resolved_date"])
        w.writerows(tickets)

    print(f"Generated CSVs:")
    print(f"  employees.csv: {len(employees)} rows")
    print(f"  sales.csv: {len(sales)} rows")
    print(f"  tickets.csv: {len(tickets)} rows")


# ─────────────────────────────────────────────
# STEP 2: Load CSVs into PostgreSQL
# ─────────────────────────────────────────────

def load_csvs_to_postgres():
    """Dynamically read CSVs and create tables."""
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    for csv_file in sorted(os.listdir("data")):
        if not csv_file.endswith(".csv"):
            continue

        table_name = csv_file.replace(".csv", "")
        filepath = os.path.join("data", csv_file)

        with open(filepath, "r") as f:
            reader = csv.reader(f)
            headers = [h.strip().lower().replace(" ", "_") for h in next(reader)]
            rows = list(reader)

        # Infer types
        col_types = []
        for col_idx in range(len(headers)):
            samples = [r[col_idx] for r in rows[:20] if col_idx < len(r) and r[col_idx]]
            is_numeric = False
            if samples:
                is_numeric = all(
                    s.replace(".", "").isdigit() and s.count(".") <= 1
                    for s in samples
                    )
            col_types.append("NUMERIC" if is_numeric else "TEXT")

        # Create table
        cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
        cols_sql = ", ".join(f'"{h}" {t}' for h, t in zip(headers, col_types))
        cursor.execute(f"CREATE TABLE {table_name} ({cols_sql})")

        # Insert data
        placeholders = ", ".join(["%s"] * len(headers))
        for row in rows:
            values = [None if v == "" else v for v in row[:len(headers)]]
            cursor.execute(f"INSERT INTO {table_name} VALUES ({placeholders})", values)

        print(f"  {table_name}: {len(rows)} rows loaded")

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# STEP 3: Extract PDF, chunk, embed, store in pgvector
# ─────────────────────────────────────────────

def load_pdf_to_pgvector(pdf_path):
    """
    Extract text from PDF → chunk → embed → store in PostgreSQL.
    
    This replaces ChromaDB entirely. 
    The document_chunks table has a VECTOR column for embeddings.
    We search it with SQL using the <=> operator.
    """
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    # Enable pgvector
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create chunks table with vector column
    cursor.execute("DROP TABLE IF EXISTS document_chunks CASCADE")
    cursor.execute("""
        CREATE TABLE document_chunks (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            embedding VECTOR(1536),
            source TEXT,
            page INTEGER,
            chunk_index INTEGER
        )
    """)
    conn.commit()

    # Extract text from PDF
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n\n"
    doc.close()

    # Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(full_text)
    print(f"  Extracted {len(chunks)} chunks from {pdf_path}")

    # Embed and store
    print(f"  Embedding and storing in pgvector...")
    for i, chunk_text in enumerate(chunks):
        embedding = client.embeddings.create(
            model="text-embedding-3-small", input=chunk_text
        ).data[0].embedding

        # Convert embedding list to pgvector format string
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        cursor.execute(
            """INSERT INTO document_chunks (text, embedding, source, page, chunk_index)
               VALUES (%s, %s, %s, %s, %s)""",
            (chunk_text, embedding_str, os.path.basename(pdf_path), 0, i),
        )

    conn.commit()

    # Create an index for faster search (important for production)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS chunks_embedding_idx
        ON document_chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 5)
    """)
    conn.commit()
    conn.close()

    print(f"  Stored {len(chunks)} chunks with embeddings in pgvector")


# ─────────────────────────────────────────────
# STEP 4: Create memory table
# ─────────────────────────────────────────────

def create_memory_table():
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS conversation_memory CASCADE")
    cursor.execute("""
        CREATE TABLE conversation_memory (
            id SERIAL PRIMARY KEY,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            agent_used TEXT,
            tools_called TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print("  conversation_memory table created")


# ─────────────────────────────────────────────
# STEP 5: Verify everything
# ─────────────────────────────────────────────

def verify():
    """Show what's in the database."""
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    print(f"\nDATABASE SUMMARY:")
    print("=" * 50)

    # List all tables with row counts
    cursor.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' ORDER BY table_name
    """)
    for (table,) in cursor.fetchall():
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count} rows")

    # Test vector search
    print(f"\nVECTOR SEARCH TEST:")
    test_query = "What is the company revenue?"
    test_emb = client.embeddings.create(
        model="text-embedding-3-small", input=test_query
    ).data[0].embedding
    emb_str = "[" + ",".join(str(x) for x in test_emb) + "]"

    cursor.execute(f"""
        SELECT text, embedding <=> '{emb_str}' AS distance
        FROM document_chunks
        ORDER BY embedding <=> '{emb_str}'
        LIMIT 2
    """)
    print(f"  Query: '{test_query}'")
    for text, dist in cursor.fetchall():
        print(f"  [{dist:.4f}] {text[:80]}...")

    # Test SQL query
    print(f"\nSQL QUERY TEST:")
    cursor.execute('SELECT "department", COUNT(*) FROM employees GROUP BY "department" ORDER BY COUNT(*) DESC')
    print(f"  Employees by department:")
    for dept, count in cursor.fetchall():
        print(f"    {dept}: {count}")

    conn.close()


# ─────────────────────────────────────────────
# RUN EVERYTHING
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("SETTING UP ALL DATA IN POSTGRESQL")
    print("=" * 50)

    print("\n1. Generating CSVs...")
    generate_csvs()

    print("\n2. Loading CSVs into PostgreSQL...")
    load_csvs_to_postgres()

    print("\n3. Loading PDF into pgvector...")
    load_pdf_to_pgvector("novatech_q3_report.pdf")

    print("\n4. Creating memory table...")
    create_memory_table()

    print("\n5. Verifying...")
    verify()

    print(f"""
{'='*50}
DONE! Everything is in ONE PostgreSQL database:
  - employees, sales, tickets (from CSVs → SQL queries)
  - document_chunks (from PDF → vector search)
  - conversation_memory (for agent memory)
  
No ChromaDB. No separate vector DB. Just PostgreSQL + pgvector.
{'='*50}
""")