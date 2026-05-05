"""
tools/sql_tool.py — PostgreSQL query execution with safety.
"""

import psycopg2
from config.settings import DATABASE_URL


def get_schema(tables_filter=None):
    """
    Discover tables, columns, types, sample data, and distinct values.
    The agent MUST call this before writing SQL.
    """
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()

    # Only show Olist tables (skip old project tables)
    olist_tables = [
        "olist_customers_dataset", "olist_orders_dataset",
        "olist_order_items_dataset", "olist_order_payments_dataset",
        "olist_order_reviews_dataset", "olist_products_dataset",
        "olist_sellers_dataset", "product_category_name_translation",
    ]

    if tables_filter:
        olist_tables = [t for t in olist_tables if t in tables_filter]

    cursor.execute("""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position
    """)
    all_cols = cursor.fetchall()

    schema = {}
    for table, col, dtype in all_cols:
        if table not in olist_tables:
            continue
        if table not in schema:
            schema[table] = []
        schema[table].append(f"{col} ({dtype})")

    output = "DATABASE SCHEMA:\n\n"
    for table, cols in schema.items():
        output += f"TABLE: {table}\n  {', '.join(cols)}\n"

        # Sample 2 rows
        try:
            cursor.execute(f"SELECT * FROM {table} LIMIT 2")
            col_names = [d[0] for d in cursor.description]
            for row in cursor.fetchall():
                output += f"  Sample: {dict(zip(col_names, row))}\n"
        except:
            pass

        # Distinct values for text columns with <= 20 unique values
        for col_info in cols:
            col_name = col_info.split(" (")[0]
            dtype = col_info.split("(")[1].rstrip(")")
            if dtype == "text":
                try:
                    cursor.execute(f'SELECT COUNT(DISTINCT "{col_name}") FROM {table}')
                    distinct_count = cursor.fetchone()[0]
                    if distinct_count <= 20:
                        cursor.execute(f'SELECT DISTINCT "{col_name}" FROM {table} LIMIT 20')
                        vals = [r[0] for r in cursor.fetchall() if r[0]]
                        output += f"  Distinct {col_name}: {vals}\n"
                except:
                    pass

        output += "\n"

    conn.close()
    return output


def execute_query(sql):
    """
    Execute a read-only SQL query with safety checks.
    Returns formatted results.
    """
    sql_upper = sql.upper().strip()

    blocked = ["DROP", "DELETE", "UPDATE", "ALTER", "INSERT", "TRUNCATE", "CREATE"]
    for kw in blocked:
        if kw in sql_upper:
            return f"ERROR: {kw} not allowed. Read-only access."

    if not sql_upper.startswith("SELECT"):
        return "ERROR: Only SELECT queries allowed."

    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SET statement_timeout = '10000'")
        cursor.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "Query returned no results."

        result = f"Columns: {columns}\nRows: {len(rows)}\n\n"
        for row in rows[:25]:
            result += str(dict(zip(columns, row))) + "\n"
        if len(rows) > 25:
            result += f"\n... and {len(rows) - 25} more rows"

        return result

    except Exception as e:
        return f"SQL Error: {e}"