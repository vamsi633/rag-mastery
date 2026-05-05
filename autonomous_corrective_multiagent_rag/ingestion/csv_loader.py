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
    """Load ONE CSV using PostgreSQL COPY — much faster than INSERT."""
    table_name = os.path.basename(filepath).replace(".csv", "").replace("-", "_")
    cursor = conn.cursor()

    print(f"  Loading {table_name}...", end=" ", flush=True)

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        headers = [h.strip().lower().replace(" ", "_") for h in next(reader)]
        rows = list(reader)

    if not rows:
        print("empty, skipping")
        return conn

    # Infer types
    col_types = []
    for col_idx in range(len(headers)):
        samples = [r[col_idx] for r in rows[:100] if col_idx < len(r) and r[col_idx]]
        col_types.append(infer_pg_type(samples))

    # Create table
    cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
    cols_sql = ", ".join(f'"{h}" {t}' for h, t in zip(headers, col_types))
    cursor.execute(f"CREATE TABLE {table_name} ({cols_sql})")
    conn.commit()

    # Use COPY for fast bulk loading
    import io

    # Build a TSV string in memory
    buffer = io.StringIO()
    for row in rows:
        values = row[:len(headers)]
        values += [""] * (len(headers) - len(values))
        # Escape tabs and newlines, replace empty with \N (NULL)
        cleaned = []
        for v in values:
            if v == "" or v is None:
                cleaned.append("\\N")
            else:
                cleaned.append(v.replace("\t", " ").replace("\n", " ").replace("\\", "\\\\"))
        buffer.write("\t".join(cleaned) + "\n")

    buffer.seek(0)

    try:
        cursor.copy_expert(
            f"""COPY {table_name} ({', '.join(f'"{h}"' for h in headers)}) 
                FROM STDIN WITH (FORMAT text, NULL '\\N')""",
            buffer,
        )
        conn.commit()
        print(f"{len(rows):,} rows ✅")
    except Exception as e:
        conn.rollback()
        print(f"COPY failed ({e}), falling back to batch INSERT...")

        # Fallback to batch insert
        placeholders = ", ".join(["%s"] * len(headers))
        inserted = 0
        batch_size = 500

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
                print(f"    {inserted:,}/{len(rows):,} rows...", flush=True)
            except Exception as e2:
                conn.rollback()
                print(f"    Batch error at {i}: {e2}")
                conn = psycopg2.connect(DATABASE_URL)
                cursor = conn.cursor()
                continue

        print(f"  {table_name}: {inserted:,} rows ✅")

    return conn

def load_all_csvs(data_dir="data"):
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
        conn = load_csv(os.path.join(data_dir, csv_file), conn)

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
    