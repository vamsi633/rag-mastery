"""
Verify pgvector is available on your Neon database.
"""

from dotenv import load_dotenv
import psycopg2
import os

load_dotenv()
conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cursor = conn.cursor()

# Enable pgvector
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
conn.commit()

# Quick test: create table with vector column, search it
cursor.execute("DROP TABLE IF EXISTS vector_test")
cursor.execute("""
    CREATE TABLE vector_test (
        id SERIAL PRIMARY KEY,
        text TEXT,
        embedding VECTOR(3)
    )
""")

# Insert 3 vectors (3 dimensions for simplicity)
cursor.execute("INSERT INTO vector_test (text, embedding) VALUES (%s, %s)",
               ("I love dogs", "[0.9, 0.1, 0.1]"))
cursor.execute("INSERT INTO vector_test (text, embedding) VALUES (%s, %s)",
               ("I love cats", "[0.8, 0.2, 0.1]"))
cursor.execute("INSERT INTO vector_test (text, embedding) VALUES (%s, %s)",
               ("Stock market crashed", "[0.1, 0.1, 0.9]"))
conn.commit()

# Vector search: find closest to "I like puppies" = [0.85, 0.15, 0.1]
cursor.execute("""
    SELECT text, embedding <=> '[0.85, 0.15, 0.1]' AS distance
    FROM vector_test
    ORDER BY embedding <=> '[0.85, 0.15, 0.1]'
    LIMIT 3
""")

print("pgvector search results (closest to 'I like puppies'):")
for text, distance in cursor.fetchall():
    print(f"  [{distance:.4f}] {text}")

# Cleanup
cursor.execute("DROP TABLE vector_test")
conn.commit()
conn.close()
print("\npgvector is working!")