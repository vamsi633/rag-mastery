from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import fitz
import json
import os
import glob


load_dotenv()
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def ingest_pdfs(pdf_paths:list[str],chroma_path='./chroma_pdfs'):
    """
    Takes a list of PDF paths, extracts text, chunks, embeds,
    stores in ChromaDB. Each chunk knows which PDF and page it came from.
    """

    chroma=chromadb.PersistentClient(path=chroma_path)


    try:
        chroma.delete_collection("pdf_docs")
    except:
        pass

    collection=chroma.create_collection("pdf_docs",metadata={"hnsw:space":"cosine"})
    splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    chunk_id=0

    for pdf_path in pdf_paths:
        filename=os.path.basename(pdf_path)
        print(f" Processing {filename}...")

        doc=fitz.open(pdf_path)
        for page_num,page in enumerate(doc):
            text=page.get_text().strip()
            if not text:
                continue
            chunks=splitter.split_text(text)
            for chunk_text in chunks:
                embedding=client.embeddings.create(model="text-embedding-3-small",input=chunk_text).data[0].embedding

                collection.add(ids=[f"chunk_{chunk_id}"],embeddings=[embedding],documents=[chunk_text],metadatas=[{"source":filename,"page":page_num+1}])
                chunk_id+=1
        doc.close()
        print(f" ->{chunk_id} total chunks so far")
    print(f" Indexed {collection.count()} chunks from {len(pdf_paths)} PDFs")
    return collection
def load_collection(chroma_path="./chroma_pdfs"):
    try:
        chroma=chromadb.PersistentClient(path=chroma_path)
        collection=chroma.get_collection("pdf_docs")
        if collection.count()>0:
            return collection
    except:
        pass
    return None

def search_documents(query:str,collection,top_k:int=5)->str:
    """Search across ALL indexed PDFs"""
    query_emb=client.embeddings.create(
        model="text-embedding-3-small",input=query
    ).data[0].embedding

    results=collection.query(query_embeddings=[query_emb],n_results=top_k)  
    # here results will be list of lists[[chunk1,chunkb,chunkc]]

    docs=results["documents"][0] #we only asked one question so results which is list of list contains answers for no of questions we onlt asked one question , so answer chunks will be at 0th position of list of lists
    distances=results["distances"][0]
    metadatas=results["metadatas"][0]

    output=""

    for i,(doc,dist,meta) in enumerate(zip(docs,distances,metadatas)):
        source=meta["source"]
        page=meta["page"]
        relevance=round(1-dist,2)
        output += f"[Result {i+1} | {source} page {page} | relevance:{relevance}]\n{doc}\n\n"

    return output

def list_sources(collection) -> str:
    """Show what documents are indexed."""
    all_meta = collection.get()["metadatas"]
    sources = {}
    for meta in all_meta:
        src = meta["source"]
        if src not in sources:
            sources[src] = {"pages": set(), "chunks": 0}
        sources[src]["pages"].add(meta["page"])
        sources[src]["chunks"] += 1

    output = "Indexed documents:\n"
    for src, info in sources.items():
        output += f"  {src}: {info['chunks']} chunks, {len(info['pages'])} pages\n"
    return output


def calculate(expression: str) -> str:
    try:
        allowed = set("0123456789.+-*/() .")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters"
        return str(round(eval(expression, {"__builtins__": {}}, {}), 2))
    except Exception as e:
        return f"Error: {e}"
    

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search across all company documents (reports, handbooks, policies). Returns relevant text with source file and page number. Search for ONE topic at a time for best results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Focused search query about one topic"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sources",
            "description": "Show what documents are available to search. Call this if you want to know what sources you have access to.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Do math. Use for percentages, totals, comparisons.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"}
                },
                "required": ["expression"],
            },
        },
    },
]
SYSTEM_PROMPT = """You are a document research assistant for NovaTech Inc.
You have access to company documents including reports and employee handbooks.

RULES:
- Search for ONE topic at a time — multiple focused searches beat one broad search
- For complex questions, search multiple times for different aspects
- Always cite the source document and page number
- Use the calculate tool for any math
- If you need to know what documents are available, use list_sources
- If information isn't found in the documents, say so clearly"""


def build_tool_functions(chroma_path="./chroma_pdfs"):
    """Tools look up the collection fresh each time — no stale references."""
    def get_collection():
        chroma = chromadb.PersistentClient(path=chroma_path)
        return chroma.get_collection("pdf_docs")
    return {
        "search_documents":lambda query,**kw:search_documents(query,get_collection()),
        "list_sources":lambda **kw:list_sources(get_collection()),
         "calculate":lambda expression,**kw:calculate(expression),
    }
def run_agent(question, tool_functions, max_steps=8):
    messages=[
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":question},
    ]
    tools_used=[]
    print(f"\n{'='*60}")
    print(f"❓ {question}")
    print(f"{'='*60}")

    for step in range(max_steps):
        response=client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOL_DEFINITIONS,
            temperature=0
        )
        message=response.choices[0].message

        if message.tool_calls:
            messages.append(message)

            for tool_call in message.tool_calls:
                func_name=tool_call.function.name
                args=json.loads(tool_call.function.arguments)
                tools_used.append(func_name)

                print(f"\n  Step {step+1} → 🔧 {func_name}")


                for k,v in args.items():
                    print(f"    {k}:{v}")

                func=tool_functions.get(func_name)
                if func:
                    try:
                        result=func(**args)
                    except TypeError:
                        result=func()
                else:
                    result=f"Unknoen tool: {func_name}"

                preview=result[:200].replace("\n"," ")
                print(f"      ->{preview}...")

                messages.append({
                    "role":"tool",
                    "tool_call_id":tool_call.id,
                    "content":str(result)
                })
        else:
            print(f"\n done in {step+1} steps | Tools: {tools_used}")
            return message.content
    return "max steps reached"

if __name__ == "__main__":
    collection = load_collection()

    if collection is None:
        pdf_files = glob.glob("*.pdf")
        if not pdf_files:
            print("No PDF files found!")
            exit()
        print(f"Found {len(pdf_files)} PDFs: {pdf_files}")
        collection = ingest_pdfs(pdf_files)
    else:
        print(f"Loaded existing index: {collection.count()} chunks")

    # Tools look up collection fresh — never goes stale
    tool_functions = build_tool_functions()

    print(f"\n🤖 NovaTech Document Assistant (PDF-only RAG)")
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
        if question.lower() == "reindex":
            pdf_files = glob.glob("*.pdf")
            collection = ingest_pdfs(pdf_files)
            print("Reindex complete!")
            continue

        answer = run_agent(question, tool_functions)
        print(f"\n💬 {answer}\n")