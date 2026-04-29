import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import client,index,EMBEDDING_MODEL,CHUNK_SIZE,CHUNK_OVERLAP
import os


def extract_text(pdf_path):
    doc=fitz.open(pdf_path)
    pages=[]
    for page_num,page in enumerate(doc):
        text=page.get_text().strip()
        if text and len(text)>50:
            pages.append({
                "text":text,
                "page":page_num+1,
                "source":os.path.basename(pdf_path)
            })
    doc.close()
    print(f" Extracted text from {len(pages)} pages")
    return pages

def chunk_pages(pages):
    splitter=RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP)

    chunks=[]
    for page_data in pages:
        page_chunks=splitter.split_text(page_data["text"])
        for chunk_text in page_chunks:
            chunks.append({
                "text":chunk_text,
                "page":page_data["text"],
                "source":page_data["source"]
            })
    print(f"  Created {len(chunks)} chunks") 
    return chunks

def embed_and_store(chunks,batch_size=50):
    """
    Embed chunks and upsert to Pinecone in batches.

    WHY BATCHES:
    - OpenAI embedding API has rate limits
    - Pinecone recommends batches of 100 max
    - Smaller batches = more reliable, less memory
    """
    # Clear existing vectors
    try:
        index.delete(delete_all=True)
        print("  Cleared existing Pinecone vectors")
    except:
        pass

    total=len(chunks)
    
    for i in range(0,total,batch_size):
        batch_chunks=chunks[i:i+batch_size]

        texts=[c["text"] for c in batch_chunks]
        response=client.embeddings.create(
            model=EMBEDDING_MODEL,input=texts
        )

        vectors=[]
        for j,(chunk,emb_data) in enumerate(zip(batch_chunks,response.data)):
            vectors.append({
                "id":f"chunk_{i+j}",
                "values":emb_data.embedding,
                "metadata":{
                    "text":chunk["text"],
                    "source":chunk["source"],
                    "page":chunk["page"]
                },
            })
        index.upsert(vectors=vectors)
        print(f"Upserted batch {i//batch_size+1}/{(total+batch_size-1)//batch_size} ({len(vectors)} vectors)")
    stats=index.describe_index_stats()
    print(f" Pinecone total: {stats.total_vector_count} vectors")

def ingest_pdf(pdf_path):
    print(f"\nProcessing {pdf_path}...")
    pages=extract_text(pdf_path)
    chunks=chunk_pages(pages)
    embed_and_store(chunks)

if __name__=="__main__":
    import glob
    pdfs=glob.glob("data/*.pdf")
    for pdf in pdfs:
        ingest_pdf(pdf)