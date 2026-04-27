import fitz
doc = fitz.open("novatech_handbook.pdf")
for i, page in enumerate(doc):
    text = page.get_text().strip()
    print(f"Page {i+1}: {len(text)} chars")
    print(f"  Preview: {text[:100]}...")
    print()
doc.close()