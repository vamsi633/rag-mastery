from dotenv import load_dotenv
from pinecone import Pinecone
import os

load_dotenv()

pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index=pc.Index(os.getenv("PINECONE_INDEX"))

stats=index.describe_index_stats()
print(f"Connected to Pinecone!")
print(f"Index: {os.getenv('PINECONE_INDEX')}")
print(f"Dimensions: {stats.dimension}")
print(f"Total vectors: {stats.total_vector_count}")