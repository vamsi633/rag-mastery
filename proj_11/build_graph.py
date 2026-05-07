"""
Project 11 — Step 1: Build a Knowledge Graph

Extract entities and relationships from our documents,
then store them in Neo4j.

Two approaches:
1. MANUAL: We define entities and relationships (what we do first)
2. LLM-EXTRACTED: LLM reads documents and extracts them automatically

We'll do both so you understand the difference.
"""

from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase
from pinecone import Pinecone
import json
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))


# ─────────────────────────────────────────────
# STEP 1: Clear existing graph
# ─────────────────────────────────────────────

def clear_graph():
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    print("  Cleared existing graph")


# ─────────────────────────────────────────────
# STEP 2: LLM-based entity extraction from documents
# ─────────────────────────────────────────────

def extract_entities_from_chunk(chunk_text: str) -> dict:
    """
    Ask the LLM to extract entities and relationships from text.
    
    This is how production knowledge graphs are built:
    - Feed document chunks to an LLM
    - LLM identifies entities (people, companies, technologies, products)
    - LLM identifies relationships (works_at, uses, manages, built_with)
    - Store in Neo4j
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Extract entities and relationships from this text.

Entity types: Person, Company, Product, Technology, Region, Department, Concept

Relationship types: WORKS_AT, MANAGES, USES, BUILT_WITH, LOCATED_IN, 
PART_OF, COMPETES_WITH, APPLIES_TO, INVESTED_IN, PARTNERS_WITH

Respond with JSON:
{
    "entities": [
        {"name": "...", "type": "Person/Company/Product/etc", "properties": {"role": "...", "description": "..."}}
    ],
    "relationships": [
        {"from": "entity_name", "to": "entity_name", "type": "USES/MANAGES/etc", "properties": {}}
    ]
}

Only extract what is EXPLICITLY stated. Don't infer.
Keep entity names consistent (always use full names)."""
            },
            {"role": "user", "content": chunk_text},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ─────────────────────────────────────────────
# STEP 3: Store in Neo4j
# ─────────────────────────────────────────────

def store_entity(session, name: str, entity_type: str, properties: dict = None):
    """Create or merge a node in Neo4j."""
    props = properties or {}
    props_str = ", ".join(f'n.{k} = "{v}"' for k, v in props.items() if v)

    query = f"""
        MERGE (n:{entity_type} {{name: $name}})
        {f'SET {props_str}' if props_str else ''}
        RETURN n
    """
    session.run(query, name=name)


def store_relationship(session, from_name: str, to_name: str, rel_type: str, properties: dict = None):
    """Create a relationship between two nodes."""
    query = f"""
        MATCH (a {{name: $from_name}})
        MATCH (b {{name: $to_name}})
        MERGE (a)-[r:{rel_type}]->(b)
        RETURN r
    """
    session.run(query, from_name=from_name, to_name=to_name)


def store_extraction(entities_and_rels: dict):
    """Store extracted entities and relationships in Neo4j."""
    with driver.session() as session:
        # Store entities
        for entity in entities_and_rels.get("entities", []):
            name = entity.get("name", "")
            etype = entity.get("type", "Thing")
            props = entity.get("properties", {})
            if name:
                store_entity(session, name, etype, props)

        # Store relationships
        for rel in entities_and_rels.get("relationships", []):
            from_name = rel.get("from", "")
            to_name = rel.get("to", "")
            rel_type = rel.get("type", "RELATED_TO")
            if from_name and to_name:
                store_relationship(session, from_name, to_name, rel_type)


# ─────────────────────────────────────────────
# STEP 4: Build the graph from Pinecone chunks
# ─────────────────────────────────────────────

def build_graph_from_documents(max_chunks=30):
    """
    Read chunks from Pinecone → extract entities with LLM → store in Neo4j.
    
    We limit to 30 chunks to save API costs.
    In production, you'd process all chunks in batch.
    """
    print("\n  Fetching chunks from Pinecone...")

    # Fetch chunks using a dummy query (gets diverse results)
    queries = [
        "technology companies and products",
        "AI and machine learning applications",
        "business strategy and market trends",
        "people and organizations",
        "cloud computing and software",
    ]

    seen_ids = set()
    all_chunks = []

    for q in queries:
        emb = client.embeddings.create(
            model="text-embedding-3-small", input=q
        ).data[0].embedding

        results = index.query(vector=emb, top_k=8, include_metadata=True)

        for match in results["matches"]:
            if match["id"] not in seen_ids:
                seen_ids.add(match["id"])
                all_chunks.append(match["metadata"]["text"])

    all_chunks = all_chunks[:max_chunks]
    print(f"  Processing {len(all_chunks)} unique chunks...")

    total_entities = 0
    total_rels = 0

    for i, chunk in enumerate(all_chunks):
        print(f"  Chunk {i+1}/{len(all_chunks)}...", end=" ", flush=True)

        extraction = extract_entities_from_chunk(chunk)
        entities = extraction.get("entities", [])
        rels = extraction.get("relationships", [])

        store_extraction(extraction)

        total_entities += len(entities)
        total_rels += len(rels)
        print(f"→ {len(entities)} entities, {len(rels)} relationships")

    return total_entities, total_rels


# ─────────────────────────────────────────────
# STEP 5: Verify the graph
# ─────────────────────────────────────────────

def show_graph_stats():
    """Show what's in the graph."""
    with driver.session() as session:
        # Count nodes by type
        result = session.run("""
            MATCH (n)
            RETURN labels(n)[0] AS type, COUNT(n) AS count
            ORDER BY count DESC
        """)
        print("\n  Nodes by type:")
        for record in result:
            print(f"    {record['type']}: {record['count']}")

        # Count relationships by type
        result = session.run("""
            MATCH ()-[r]->()
            RETURN type(r) AS type, COUNT(r) AS count
            ORDER BY count DESC
        """)
        print("\n  Relationships by type:")
        for record in result:
            print(f"    {record['type']}: {record['count']}")

        # Show some example paths
        result = session.run("""
            MATCH (a)-[r]->(b)
            RETURN a.name AS from, type(r) AS rel, b.name AS to
            LIMIT 10
        """)
        print("\n  Sample relationships:")
        for record in result:
            print(f"    {record['from']} --[{record['rel']}]--> {record['to']}")


# ─────────────────────────────────────────────
# RUN IT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("BUILDING KNOWLEDGE GRAPH")
    print("=" * 50)

    print("\nStep 1: Clearing old graph...")
    clear_graph()

    print("\nStep 2: Extracting entities from documents...")
    total_e, total_r = build_graph_from_documents(max_chunks=20)
    print(f"\n  Total: {total_e} entities, {total_r} relationships extracted")

    print("\nStep 3: Graph statistics...")
    show_graph_stats()

    driver.close()
    print("\nDone! Graph is ready in Neo4j.")