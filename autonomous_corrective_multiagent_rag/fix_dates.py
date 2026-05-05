"""
Fix date columns in Olist tables — convert TEXT to TIMESTAMP/DATE.
Run ONCE after ingestion.
"""

from dotenv import load_dotenv
import psycopg2
import os

load_dotenv()
conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cursor = conn.cursor()

# All date/timestamp columns in Olist dataset
date_columns = [
    ("olist_orders_dataset", "order_purchase_timestamp", "TIMESTAMP"),
    ("olist_orders_dataset", "order_approved_at", "TIMESTAMP"),
    ("olist_orders_dataset", "order_delivered_carrier_date", "TIMESTAMP"),
    ("olist_orders_dataset", "order_delivered_customer_date", "TIMESTAMP"),
    ("olist_orders_dataset", "order_estimated_delivery_date", "TIMESTAMP"),
    ("olist_order_items_dataset", "shipping_limit_date", "TIMESTAMP"),
    ("olist_order_reviews_dataset", "review_creation_date", "TIMESTAMP"),
    ("olist_order_reviews_dataset", "review_answer_timestamp", "TIMESTAMP"),
]

for table, col, target_type in date_columns:
    try:
        temp_col = f"{col}_temp"
        cursor.execute(f'ALTER TABLE {table} ADD COLUMN IF NOT EXISTS "{temp_col}" {target_type}')
        cursor.execute(f"""
            UPDATE {table} SET "{temp_col}" = 
            CASE 
                WHEN "{col}" IS NOT NULL AND "{col}" != '' 
                THEN "{col}"::{target_type}
                ELSE NULL 
            END
        """)
        cursor.execute(f'ALTER TABLE {table} DROP COLUMN "{col}"')
        cursor.execute(f'ALTER TABLE {table} RENAME COLUMN "{temp_col}" TO "{col}"')
        conn.commit()
        print(f"  ✅ {table}.{col} → {target_type}")
    except Exception as e:
        conn.rollback()
        print(f"  ❌ {table}.{col}: {e}")

# Also fix payment_value and price to NUMERIC if they're text
numeric_columns = [
    ("olist_order_items_dataset", "price"),
    ("olist_order_items_dataset", "freight_value"),
    ("olist_order_payments_dataset", "payment_value"),
    ("olist_order_payments_dataset", "payment_installments"),
    ("olist_order_reviews_dataset", "review_score"),
]

for table, col in numeric_columns:
    try:
        temp_col = f"{col}_temp"
        cursor.execute(f'ALTER TABLE {table} ADD COLUMN IF NOT EXISTS "{temp_col}" NUMERIC')
        cursor.execute(f"""
            UPDATE {table} SET "{temp_col}" = 
            CASE 
                WHEN "{col}" IS NOT NULL AND "{col}" != '' 
                THEN "{col}"::NUMERIC
                ELSE NULL 
            END
        """)
        cursor.execute(f'ALTER TABLE {table} DROP COLUMN "{col}"')
        cursor.execute(f'ALTER TABLE {table} RENAME COLUMN "{temp_col}" TO "{col}"')
        conn.commit()
        print(f"  ✅ {table}.{col} → NUMERIC")
    except Exception as e:
        conn.rollback()
        print(f"  ❌ {table}.{col}: {e}")

# Verify
print("\nVerifying key columns:")
cursor.execute("""
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = 'public'
    AND (column_name LIKE '%date%' OR column_name LIKE '%timestamp%' 
         OR column_name IN ('price', 'freight_value', 'payment_value', 'review_score'))
    ORDER BY table_name, column_name
""")
for table, col, dtype in cursor.fetchall():
    print(f"  {table}.{col} → {dtype}")

conn.close()
print("\nDone! Date and numeric columns fixed.")