import psycopg2
import csv
import os
from config.settings import DATABASE_URL

SKIP_FILES={"olist_geolocation_dataset.csv"}

def infer_pg_type(values):
    """Guess PostgreSQL column type from sample values."""
    if not values:
        return "TEXT"
    for v in values:
        if v == "":
            continue
        try:
            int(v)
            return "INTEGER"
        except ValueError:
            pass
        try:
            float(v)
            return "NUMERIC"
        except ValueError:
            return "TEXT"
    return "TEXT"

def load_csv(filepath, conn):
    """Load ONE CSV into PostgreSQL with batch commits."""
    table_name = os.path.basename(filepath).replace(".csv", "").replace("-", "_")
    cursor = conn.cursor()

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        headers = [h.strip().lower().replace(" ", "_") for h in next(reader)]
        rows = list(reader)

    if not rows:
        print(f"  Skipping {table_name} — empty")
        return

    # Infer types from first 100 rows
    col_types = []
    for col_idx in range(len(headers)):
        samples = [r[col_idx] for r in rows[:100] if col_idx < len(r) and r[col_idx]]
        col_types.append(infer_pg_type(samples))

    # Create table
    cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
    cols_sql = ", ".join(f'"{h}" {t}' for h, t in zip(headers, col_types))
    cursor.execute(f"CREATE TABLE {table_name} ({cols_sql})")
    conn.commit()

    # Insert in small batches with commit after each
    placeholders = ", ".join(["%s"] * len(headers))
    batch_size = 500
    inserted = 0

    for i in range(0, len(rows), batch_size):
        batch = []
        for row in rows[i:i + batch_size]:
            values = [None if v == "" else v for v in row[:len(headers)]]
            values += [None] * (len(headers) - len(values))
            batch.append(values)

        try:
            cursor.executemany(
                f"INSERT INTO {table_name} VALUES ({placeholders})", batch
            )
            conn.commit()
            inserted += len(batch)
        except Exception as e:
            conn.rollback()
            print(f"    Error at batch {i}: {e}")
            # Reconnect if connection dropped
            conn = psycopg2.connect(DATABASE_URL)
            cursor = conn.cursor()
            continue

    print(f"  {table_name}: {inserted:,} rows, {len(headers)} columns")
    return conn  # return conn in case it was reconnected

def load_all_csvs(data_dir="data"):
    """Load all CSVs from data/ folder into PostgreSQL."""
    conn = psycopg2.connect(DATABASE_URL)

    csv_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith(".csv") and f not in SKIP_FILES
    ])

    if not csv_files:
        print("No CSV files found!")
        return

    print(f"Loading {len(csv_files)} CSVs into PostgreSQL...")
    for csv_file in csv_files:
        result = load_csv(os.path.join(data_dir, csv_file), conn)
        if result:
            conn = result  # use reconnected connection if needed

    # Create memory table
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

    # Print summary
    cursor.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' ORDER BY table_name
    """)
    print(f"\nDatabase summary:")
    for (table,) in cursor.fetchall():
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table}: {count:,} rows")

    conn.close()

if __name__ == "__main__":
    load_all_csvs()
    