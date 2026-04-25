from dotenv import load_dotenv
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import os

load_dotenv()

client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
handbook = """
Chapter 1: Company Overview

TechCorp was founded in 2015 in San Francisco by Jane Smith and Robert Chen. The company specializes in cloud infrastructure and developer tools. As of 2024, TechCorp has over 2,000 employees across 5 offices worldwide including San Francisco, New York, London, Tokyo, and Bangalore.

Our mission is to make cloud computing accessible to every developer. We believe in open source, transparency, and developer experience above all else.

Chapter 2: Employment Policies

2.1 Work Schedule
Standard working hours are 9 AM to 5 PM, Monday through Friday. Flexible scheduling is available with manager approval. Core hours when all employees must be available are 10 AM to 3 PM.

Remote work is permitted up to 3 days per week. Fully remote positions are available for engineering roles with director approval. All remote employees must have a reliable internet connection and a dedicated workspace.

2.2 Compensation
Salaries are reviewed annually in March. Merit increases range from 2% to 8% based on performance ratings. Promotions may include salary adjustments of 10% to 20%.

Stock options are granted to all full-time employees. The standard vesting schedule is 4 years with a 1-year cliff. Employees receive their first grant on their start date and may receive additional grants based on performance.

2.3 Time Off
Employees receive 20 days of paid time off (PTO) per year. PTO accrues monthly at 1.67 days per month. Unused PTO carries over up to a maximum of 30 days.

Sick leave provides 10 days per year and does not roll over. A doctor's note is required for absences exceeding 3 consecutive days. Mental health days are included in sick leave.

There are 11 paid company holidays per year. If a holiday falls on a weekend, it is observed on the nearest weekday.

Chapter 3: Benefits

3.1 Health Insurance
TechCorp offers three health plan tiers: Basic, Standard, and Premium. Basic covers medical only at $50/month. Standard adds dental and vision at $120/month. Premium includes everything plus mental health coverage at $200/month.

The company covers 80% of the employee premium and 50% of dependent premiums. Open enrollment occurs annually in November.

3.2 Retirement
The company offers a 401(k) plan with employer matching up to 5% of salary. Employees are eligible after 90 days of employment. Vesting of employer contributions follows a 3-year graded schedule: 33% after year 1, 66% after year 2, and 100% after year 3.

3.3 Other Benefits
Education reimbursement up to $5,000 per year for job-related courses and certifications. Gym membership reimbursement up to $100 per month. Commuter benefits provide pre-tax transit and parking deductions. New parents receive 16 weeks of paid parental leave regardless of gender.

Chapter 4: Equipment and Security

All employees receive a laptop (MacBook Pro or ThinkPad, employee's choice) and a 27-inch monitor on their first day. Equipment must be returned within 5 business days of the last day of employment.

Company devices must have full-disk encryption enabled and use the corporate VPN when accessing internal systems remotely. Passwords must be at least 12 characters and changed every 90 days. Two-factor authentication is required for all company accounts.

Employees should report security incidents to security@techcorp.com within 1 hour of discovery. The security team operates 24/7 and will respond to all reports within 30 minutes.
"""

splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50,separators=["\n\n", "\n", ". ", " "],)

chunks=splitter.split_text(handbook)

chroma=chromadb.Client()
collection=chroma.create_collection("handbook",metadata={"hnsw:space":"cosine"})

for i,chunk in enumerate(chunks):
    embedding=client.embeddings.create(
        model="text-embedding-3-small",
        input=chunk
    ).data[0].embedding

    collection.add(
        ids=[f"chunks_{i}"],
        embeddings=[embedding],
        documents=[chunk],
        metadatas=[{"chunk_index":1,"char_count":len(chunk)}]
    )
print(f" Scored {collection.count()} chunks with embeddings")

def rag_query(question,top_k=3):
    query_emb=client.embeddings.create(
        model="text-embedding-3-small",input=question
    ).data[0].embedding

    results=collection.query(query_embeddings=[query_emb],n_results=top_k)

    retrived_chunks=results["documents"][0]
    distances=results["distances"][0]


    
    context="\n\n".join(f"[Source {i+1}]: {c}" for i,c in enumerate(retrived_chunks))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Answer based ONLY on the context. If the answer isn't there, say so. Be concise. Cite sources like [Source 1]."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )

    return response.choices[0].message.content

questions = [
    "How many vacation days do I get per year?",
    "What laptop options are available?",
    "How does the 401k matching work?",
    "Can I work remotely full time?",
    "What is the password policy?",
    "Who founded the company and when?",
    "What is the best programming language?",  # NOT in the document
]

for q in questions:
    answer = rag_query(q)
    print(q)
    print(f"💬 Answer: {answer}\n")