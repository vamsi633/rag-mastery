"""
Project 3: Multimodal RAG — Complete Pipeline

PDF with text + charts → Extract all → Caption images → 
Chunk everything → Store in ChromaDB → Query across text AND images
"""

from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import fitz  # PyMuPDF
import base64
import os
import json

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─────────────────────────────────────────────
# STEP 1: EXTRACT text + images from PDF
# ─────────────────────────────────────────────

def extract_from_pdf(pdf_path):
    """
    Extract all content from a PDF.
    Returns text per page and saved image paths.
    In production, this handles scanned PDFs with OCR too.
    """
    doc = fitz.open(pdf_path)
    os.makedirs("extracted_images", exist_ok=True)

    pages_text = []
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Extract text
        text = page.get_text().strip()
        if text:
            pages_text.append({
                "text": text,
                "page": page_num + 1,
                "source": pdf_path,
            })

        # Extract images
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_path = f"extracted_images/page{page_num + 1}_img{img_idx + 1}.{image_ext}"
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            images.append({
                "path": image_path,
                "page": page_num + 1,
                "source": pdf_path,
            })

    doc.close()
    return pages_text, images


# ─────────────────────────────────────────────
# STEP 2: CAPTION images using GPT-4o
# ─────────────────────────────────────────────

def caption_image(image_path):
    """
    Send image to GPT-4o, get detailed text description.

    WHY DETAILED?
    The caption is what gets EMBEDDED and SEARCHED.
    If the caption says "a bar chart" — too vague.
    If it says "bar chart showing Q3 revenue: North America
    $14.2M, Europe $8.8M, Asia Pacific $11.3M" — now a query
    like "What was Asia Pacific revenue?" will match it.
    """
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Describe this image in detail for a document retrieval system. "
                        "Include ALL numbers, labels, values, trends, axis titles, "
                        "and what the image represents. Extract every piece of data visible. "
                        "Format as a structured description."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ],
        }],
        max_tokens=600,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────
# STEP 3: CHUNK text
# ─────────────────────────────────────────────

def chunk_text(pages_text, chunk_size=500, overlap=50):
    """
    Chunk all extracted text.
    Each chunk keeps metadata about which page it came from.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = []
    for page_data in pages_text:
        page_chunks = splitter.split_text(page_data["text"])
        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "type": "text",
                "page": page_data["page"],
                "source": page_data["source"],
            })

    return chunks


# ─────────────────────────────────────────────
# STEP 4: BUILD MULTIMODAL INDEX in ChromaDB
# ─────────────────────────────────────────────

def build_index(text_chunks, image_data):
    """
    Store BOTH text chunks and image captions in the same
    ChromaDB collection. They share the same embedding space.

    Text chunks: embedded directly
    Image captions: embedded as text (the caption IS the searchable content)

    Metadata tracks the type so we know what to pass to the LLM:
    - type="text" → pass the text to LLM
    - type="image" → pass the ORIGINAL IMAGE to LLM (not just caption)
    """
    chroma = chromadb.PersistentClient(path="./chroma_multimodal")

    # Delete old collection if exists
    try:
        chroma.delete_collection("multimodal_docs")
    except:
        pass

    collection = chroma.create_collection(
        "multimodal_docs",
        metadata={"hnsw:space": "cosine"},
    )

    all_entries = []

    # Add text chunks
    for i, chunk in enumerate(text_chunks):
        all_entries.append({
            "id": f"text_{i}",
            "content": chunk["text"],
            "metadata": {
                "type": "text",
                "page": chunk["page"],
                "source": chunk["source"],
            },
        })

    # Add image captions
    for i, img in enumerate(image_data):
        all_entries.append({
            "id": f"image_{i}",
            "content": img["caption"],
            "metadata": {
                "type": "image",
                "page": img["page"],
                "source": img["source"],
                "image_path": img["path"],
            },
        })

    # Embed and store everything
    print(f"  Embedding {len(all_entries)} entries...")
    for entry in all_entries:
        embedding = client.embeddings.create(
            model="text-embedding-3-small",
            input=entry["content"],
        ).data[0].embedding

        collection.add(
            ids=[entry["id"]],
            embeddings=[embedding],
            documents=[entry["content"]],
            metadatas=[entry["metadata"]],
        )

    return collection


# ─────────────────────────────────────────────
# STEP 5: MULTIMODAL RAG QUERY
# ─────────────────────────────────────────────

def multimodal_rag_query(collection, question, top_k=4):
    """
    Query that searches BOTH text and image captions.
    When an image matches, we pass the ORIGINAL IMAGE
    to GPT-4o for the final answer — not just the caption.

    This is the key:
    - Caption is used for FINDING the right image (search)
    - Original image is used for ANSWERING (generation)
    """
    # Embed the query
    query_emb = client.embeddings.create(
        model="text-embedding-3-small", input=question
    ).data[0].embedding

    # Search across text + image captions
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Show what was retrieved
    print(f"\n🔍 Query: {question}")
    print(f"   Retrieved {len(documents)} results:")
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        print(f"   [{i+1}] ({meta['type']}, page {meta['page']}, dist:{dist:.3f}) {doc[:70]}...")

    # Build the prompt — handle text and images differently
    text_context = []
    image_messages = []

    for doc, meta in zip(documents, metadatas):
        if meta["type"] == "text":
            text_context.append(f"[Text from page {meta['page']}]: {doc}")

        elif meta["type"] == "image":
            text_context.append(f"[Image description from page {meta['page']}]: {doc}")

            # Also include the ORIGINAL image for GPT-4o to see
            image_path = meta.get("image_path", "")
            if os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode()
                image_messages.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                })

    context = "\n\n".join(text_context)

    # Build message content — text + images together
    user_content = [
        {
            "type": "text",
            "text": f"""Answer the question using the context and any images provided.
If the answer comes from a chart or image, extract the exact data from it.
Cite your sources (page numbers).

Context:
{context}

Question: {question}""",
        }
    ]
    # Attach any retrieved images
    user_content.extend(image_messages)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a financial analyst assistant. Answer precisely using the provided context and images. Always cite page numbers."},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
    )

    return response.choices[0].message.content


# ─────────────────────────────────────────────
# RUN THE COMPLETE PIPELINE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    pdf_path = "novatech_q3_report.pdf"

    # STEP 1: Extract
    print("STEP 1: Extracting from PDF...")
    pages_text, images = extract_from_pdf(pdf_path)
    print(f"  Extracted text from {len(pages_text)} pages")
    print(f"  Extracted {len(images)} images")

    # STEP 2: Caption images
    print("\nSTEP 2: Captioning images with GPT-4o...")
    image_data = []
    for img in images:
        print(f"  Captioning {img['path']}...")
        caption = caption_image(img["path"])
        print(f"    Caption: {caption[:100]}...")
        image_data.append({
            "path": img["path"],
            "page": img["page"],
            "source": img["source"],
            "caption": caption,
        })

    # STEP 3: Chunk text
    print("\nSTEP 3: Chunking text...")
    text_chunks = chunk_text(pages_text)
    print(f"  Created {len(text_chunks)} text chunks")

    # STEP 4: Build index
    print("\nSTEP 4: Building ChromaDB index...")
    collection = build_index(text_chunks, image_data)
    print(f"  Indexed {collection.count()} total entries (text + images)")

    # STEP 5: Query!
    print(f"\n{'='*60}")
    print("STEP 5: Querying the Multimodal RAG")
    print(f"{'='*60}")

    questions = [
        "What was the revenue for Asia Pacific in Q3 2024?",        # in chart AND text
        "How does revenue compare to operating costs over time?",   # in line chart only
        "What is the employee headcount?",                          # in text only
        "What are the risk factors for Q4?",                        # in text only
        "Which region grew fastest by percentage?",                 # needs chart + text
        "What is the Q4 revenue projection?",                      # in text only
    ]

    for q in questions:
        answer = multimodal_rag_query(collection, q)
        print(f"\n💬 Answer: {answer}")
        print(f"{'─'*60}")