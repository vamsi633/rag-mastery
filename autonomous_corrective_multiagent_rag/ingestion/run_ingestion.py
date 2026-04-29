"""
ingestion/run_ingestion.py — Run the complete ingestion pipeline.
Usage: python -m ingestion.run_ingestion
"""

from ingestion.csv_loader import load_all_csvs
from ingestion.pdf_loader import ingest_pdf
import glob
import os


def run():
    print("=" * 60)
    print("INGESTION PIPELINE")
    print("=" * 60)

    # Step 1: CSVs → PostgreSQL
    print("\n📊 Step 1: Loading CSVs into PostgreSQL...")
    load_all_csvs("data")

    # Step 2: PDFs → Pinecone
    print("\n📄 Step 2: Loading PDFs into Pinecone...")
    pdfs = glob.glob("data/*.pdf")
    if pdfs:
        for pdf in pdfs:
            ingest_pdf(pdf)
    else:
        print("  No PDFs found in data/")

    print(f"\n{'='*60}")
    print("INGESTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run()