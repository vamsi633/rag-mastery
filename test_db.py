from dotenv import load_dotenv
import psycopg2
import os

load_dotenv()

conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cursor = conn.cursor()

cursor.execute("SELECT version();")
print("Connected to:", cursor.fetchone()[0])

cursor.execute("SELECT current_database();")
print("Database:", cursor.fetchone()[0])

conn.close()
print("Connection works!")