"""
Project 11 — Graph RAG Query System

Combines:
  Neo4j:    Relationship traversal (structured connections)
  Pinecone: Text search (detailed context)
  LLM:      Combines both into a coherent answer
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
# STEP 1: Extract entities from the question
# ─────────────────────────────────────────────

def extract_query_entities(question: str) -> list[str]:
    """
    Find which entities in the question exist in our graph.
    This gives us starting points for traversal.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Extract key entities (technologies, companies, concepts, products) 
from this question. Return JSON: {"entities": ["entity1", "entity2"]}
Only include specific named things, not generic words."""
            },
            {"role": "user", "content": question},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    return result.get("entities", [])


# ─────────────────────────────────────────────
# STEP 2: Query Neo4j for relationships
# ─────────────────────────────────────────────

def search_graph(entities: list[str]) -> str:
    """
    Find all relationships connected to the mentioned entities.
    This is where graph RAG shines — following connections.
    """
    graph_context = ""

    with driver.session() as session:
        for entity in entities:
            # Find the entity node (fuzzy match)
            result = session.run("""
                MATCH (n)
                WHERE toLower(n.name) CONTAINS toLower($name)
                RETURN n.name AS name, labels(n)[0] AS type
                LIMIT 3
            """, name=entity)

            found_nodes = list(result)

            if not found_nodes:
                continue

            for node in found_nodes:
                node_name = node["name"]
                node_type = node["type"]
                graph_context += f"\nEntity: {node_name} (type: {node_type})\n"

                # Find all relationships FROM this entity (1-2 hops)
                rels = session.run("""
                    MATCH (n {name: $name})-[r]->(target)
                    RETURN n.name AS from, type(r) AS rel, target.name AS to,
                           labels(target)[0] AS target_type
                """, name=node_name)

                for record in rels:
                    graph_context += f"  {record['from']} --[{record['rel']}]--> {record['to']} ({record['target_type']})\n"

                # Find all relationships TO this entity
                rels_in = session.run("""
                    MATCH (source)-[r]->(n {name: $name})
                    RETURN source.name AS from, type(r) AS rel, n.name AS to,
                           labels(source)[0] AS source_type
                """, name=node_name)

                for record in rels_in:
                    graph_context += f"  {record['from']} ({record['source_type']}) --[{record['rel']}]--> {record['to']}\n"

                # 2-hop: Find connections of connections
                two_hop = session.run("""
                    MATCH (n {name: $name})-[r1]->(mid)-[r2]->(target)
                    RETURN n.name AS start, type(r1) AS rel1, mid.name AS middle,
                           type(r2) AS rel2, target.name AS end
                    LIMIT 10
                """, name=node_name)

                for record in two_hop:
                    graph_context += f"  2-hop: {record['start']} --[{record['rel1']}]--> {record['middle']} --[{record['rel2']}]--> {record['end']}\n"

    return graph_context if graph_context else "No graph relationships found for these entities."


# ─────────────────────────────────────────────
# STEP 3: Search Pinecone for text context
# ─────────────────────────────────────────────

def search_documents(query: str) -> str:
    """Standard vector search for detailed text context."""
    emb = client.embeddings.create(
        model="text-embedding-3-small", input=query
    ).data[0].embedding

    results = index.query(vector=emb, top_k=4, include_metadata=True)

    context = ""
    for match in results["matches"]:
        text = match["metadata"]["text"]
        source = match["metadata"].get("source", "?")
        page = match["metadata"].get("page", "?")
        score = round(match["score"], 3)
        context += f"[{source} p{page}, score:{score}] {text}\n\n"

    return context if context else "No documents found."


# ─────────────────────────────────────────────
# STEP 4: Generate answer using BOTH sources
# ─────────────────────────────────────────────

def generate_answer(question: str, graph_context: str, text_context: str) -> str:
    """Combine graph relationships + document text into one answer."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a research analyst. Answer using BOTH sources:

1. GRAPH DATA: Shows entities and their relationships (connections, hierarchies)
   Use this for: who connects to what, what uses what, structural facts

2. DOCUMENT TEXT: Shows detailed descriptions and explanations
   Use this for: context, details, numbers, explanations

Combine both for a complete answer. Cite which source supported each claim."""
            },
            {
                "role": "user",
                "content": f"""GRAPH RELATIONSHIPS:
{graph_context}

DOCUMENT CONTEXT:
{text_context}

QUESTION: {question}"""
            },
        ],
        temperature=0,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────
# COMPLETE GRAPH RAG PIPELINE
# ─────────────────────────────────────────────

def graph_rag(question: str) -> str:
    """
    Full pipeline:
    1. Extract entities from question
    2. Search Neo4j for relationships (structure)
    3. Search Pinecone for text (detail)
    4. Combine both for answer
    """
    print(f"\n{'='*60}")
    print(f"❓ {question}")
    print(f"{'='*60}")

    # Step 1: Extract entities
    entities = extract_query_entities(question)
    print(f"\n  🏷️  Entities found: {entities}")

    # Step 2: Graph search
    print(f"\n  🕸️  Searching Neo4j...")
    graph_context = search_graph(entities)
    graph_preview = graph_context[:300].replace("\n", "\n     ")
    print(f"     {graph_preview}...")

    # Step 3: Document search
    print(f"\n  📄 Searching Pinecone...")
    text_context = search_documents(question)
    text_preview = text_context[:200].replace("\n", " ")
    print(f"     {text_preview}...")

    # Step 4: Generate
    print(f"\n  🤖 Generating answer from graph + documents...")
    answer = generate_answer(question, graph_context, text_context)

    return answer


# ─────────────────────────────────────────────
# INTERACTIVE MODE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Show what's in the graph
    with driver.session() as session:
        result = session.run("MATCH (n) RETURN COUNT(n) AS nodes")
        nodes = result.single()["nodes"]
        result = session.run("MATCH ()-[r]->() RETURN COUNT(r) AS rels")
        rels = result.single()["rels"]
    print(f"Neo4j: {nodes} nodes, {rels} relationships")

    stats = index.describe_index_stats()
    print(f"Pinecone: {stats.total_vector_count} vectors")

    print(f"\n🤖 Graph RAG System (Neo4j + Pinecone)")
    print("   Type 'quit' to exit\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        answer = graph_rag(question)
        print(f"\n💬 {answer}\n")

    driver.close()