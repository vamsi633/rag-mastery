from config.settings import client,index,EMBEDDING_MODEL,TOP_K

def search(query,top_k=TOP_K):
    """
    Semantic search over documents in Pinecone.
    Returns formatted results with source, page, and relevance.
    """

    emb=client.embeddings.create(model=EMBEDDING_MODEL,input=query).data[0].embedding

    results=index.query(vector=emb,top_k=top_k,include_metadata=True)

    if not results["matches"]:
        return "No results found."
    
    output=""

    for i,match in enumerate(results["matches"]):
        text=match["metadata"]["text"]
        source=match["metadata"].get("source","unknown")
        page=match["metadata"].get("page","?")
        score=round(match["score"],3)
        output+=f"[Result {i+1} | {source} p{page} | score:{score}]\n{text}\n\n"
    return output

def search_raw(query,top_k=TOP_K):
    emb=client.embeddings.create(
        model=EMBEDDING_MODEL,input=query
    ).data[0].embedding

    results=index.query(vector=emb,top_k=top_k,include_metadata=True)

    chunks=[]
    for match in results["matches"]:
        chunks.append({
             "text": match["metadata"]["text"],
            "source": match["metadata"].get("source", "unknown"),
            "page": match["metadata"].get("page", "?"),
            "score": round(match["score"], 3),
        })

    return chunks

