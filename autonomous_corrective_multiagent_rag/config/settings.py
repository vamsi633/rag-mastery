import os
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
client=OpenAI(api_key=OPENAI_API_KEY)
EMBEDDING_MODEL="text-embedding-3-small"
LLM_MODEL="gpt-4o-mini"


PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
PINECONE_INDEX=os.getenv("PINECONE_INDEX")

pc=Pinecone(api_key=PINECONE_API_KEY)
index=pc.Index(PINECONE_INDEX)


DATABASE_URL=os.getenv("DATABASE_URL")

CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
MAX_AGENT_STEPS=6
MAX_AUTONOMOUS_ITERATIONS=3



