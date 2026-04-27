from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz
import os
import glob

load_dotenv()
client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index=pc.Index(os.getenv("PINECONE_INDEX"))


def ingest_pdfs(pdf_paths:list[str]):
    splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    chunk_id=0
    all_vectors=[]

    for pdf_path in pdf_paths:
        filename=os.path.basename(pdf_path)
        print(f"   Processing{filename}...")

        doc=fitz.open(pdf_path)
        for page_num,page in enumerate(doc):
            text=page.get_text().strip()
            if not text:
                continue

            chunks=splitter.split_text(text)
            for chunk_text in chunks:
                embedding=client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk_text
                ).data[0].embedding

                all_vectors.append({
                    "id":f"chunk_{chunk_id}",
                    "values":embedding,
                    "metadata":{
                        "text":chunk_text, 
                        "source":filename,
                        "page":page_num+1,                   
                        },
                })
                chunk_id+=1
        doc.close()
        print(f"  {chunk_id} chunks so far")
    batch_size=100
    for i in range(0,len(all_vectors),batch_size):
        batch=all_vectors[i:i+batch_size]
        index.upsert(vectors=batch)
        print(f"  Upserted batch {i//batch_size+1}({len(batch)} vectors)")
    
    stats=index.describe_index_stats()
    print(f"\n Done! Pinecone now has {stats.total_vector_count} vectors")

if __name__=="__main__":
    pdf_files=glob.glob("*.pdf") + glob.glob("../agentic_rag/*.pdf")

    if not pdf_files:
        print("No PDFs found! Copy your PDFs here or check the path.")
        exit()

    print(f"Found {len(pdf_files)} PDFs: {[os.path.basename(f) for f in pdf_files]}")
    print("Ingesting into Pinecone...\n")

    ingest_pdfs(pdf_files)
