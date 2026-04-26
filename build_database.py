"""
Dynamically reads CSVs from data/ folder and creates PostgreSQL tables.
Drop in ANY CSV — it auto-creates the table and imports the data.
Schema is discovered at runtime, not hardcoded.
"""

from dotenv import load_dotenv
import psycopg2
import csv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
DATA_DIR = "data"


def infer_pg_type(values):
    """Guess PostgreSQL column type from sample values."""
    has_value = False
    is_int = True
    is_float = True

    for v in values:
        if v == "":
            continue
        has_value = True
        try:
            int(v)
        except ValueError:
            is_int = False
        try:
            float(v)
        except ValueError:
            is_float = False

    if not has_value:
        return "TEXT"
    if is_int:
        return "INTEGER"
    if is_float:
        return "NUMERIC"
    return "TEXT"


def import_csv_to_postgres(csv_path, conn):
    """
    Reads ANY CSV and creates a matching PostgreSQL table.

    Production ETL pattern:
    1. Read header → column names
    2. Sample rows → infer types
    3. DROP + CREATE TABLE dynamically
    4. INSERT all rows with parameterized queries (SQL injection safe)
    """
    table_name = os.path.splitext(os.path.basename(csv_path))[0]
    cursor = conn.cursor()

    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

    if not rows:
        print(f"  Skipping {csv_path} — empty file")
        return

    # Clean column names (PostgreSQL is strict about names)
    clean_headers = [h.strip().lower().replace(" ", "_") for h in headers]

    # Infer column types from first 50 rows
    col_types = []
    for col_idx in range(len(clean_headers)):
        sample_values = [row[col_idx] for row in rows[:50] if col_idx < len(row)]
        col_type = infer_pg_type(sample_values)
        col_types.append(col_type)

    # Drop and create table
    cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")

    columns_sql = ", ".join(
        f'"{h}" {t}' for h, t in zip(clean_headers, col_types)
    )
    cursor.execute(f"CREATE TABLE {table_name} ({columns_sql})")

    # Insert rows with parameterized queries (safe from SQL injection)
    placeholders = ", ".join(["%s"] * len(clean_headers))
    insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"

    for row in rows:
        # Pad or trim row to match headers
        padded = row + [""] * (len(clean_headers) - len(row))
        values = padded[:len(clean_headers)]

        # Convert empty strings to None for proper NULL handling
        values = [None if v == "" else v for v in values]

        cursor.execute(insert_sql, values)

    conn.commit()
    print(f"  {table_name}: {len(rows)} rows, {len(clean_headers)} columns")


def build_database():
    """Import all CSVs from data/ folder into PostgreSQL."""
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    # Create conversation memory table
    cursor.execute("DROP TABLE IF EXISTS conversation_memory CASCADE")
    cursor.execute("""
        CREATE TABLE conversation_memory (
            id SERIAL PRIMARY KEY,
            query TEXT NOT NULL,
            answer TEXT NOT NULL,
            agent_used TEXT NOT NULL,
            tools_called TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    # Import all CSVs
    print("Importing CSVs into PostgreSQL...")
    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])

    if not csv_files:
        print(f"  No CSV files found in {DATA_DIR}/")
        conn.close()
        return

    for csv_file in csv_files:
        import_csv_to_postgres(os.path.join(DATA_DIR, csv_file), conn)

    # Print database summary
    print(f"\nDatabase ready on Neon PostgreSQL")
    print(f"Tables:")

    cursor.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    tables = cursor.fetchall()

    for (table,) in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]

        cursor.execute(f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table}'
            ORDER BY ordinal_position
        """)
        columns = cursor.fetchall()
        col_info = [f"{name}({dtype})" for name, dtype in columns]

        print(f"  {table}: {count} rows")
        print(f"    columns: {col_info}")

    # Show sample data
    print(f"\nSample — Employees:")
    cursor.execute('SELECT "name", "department", "role", "salary" FROM employees LIMIT 5')
    for row in cursor.fetchall():
        print(f"  {row[0]} | {row[1]} | {row[2]} | ${float(row[3]):,.0f}")

    print(f"\nSample — Sales:")
    cursor.execute("""SELECT "deal_name", "amount", "region", "status"
                      FROM sales WHERE "status" = 'closed_won' LIMIT 5""")
    for row in cursor.fetchall():
        print(f"  {row[0]} | ${float(row[1]):,.0f} | {row[2]} | {row[3]}")

    print(f"\nSample — Open tickets:")
    cursor.execute("""SELECT "customer", "issue", "priority"
                      FROM tickets WHERE "status" = 'open'""")
    for row in cursor.fetchall():
        print(f"  {row[0]} | {row[1][:50]}... | {row[2]}")

    conn.close()


if __name__ == "__main__":
    build_database()
    print("\nDrop any CSV into data/ and re-run to add more tables.")