import os
import pdfplumber

CHUNKS_DIR = os.path.join(os.path.dirname(__file__), "chunks")
os.makedirs(CHUNKS_DIR, exist_ok=True)

def chunk_pdf(pdf_path, chunk_size=1000, overlap=200):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks

def run():
    from pathlib import Path
    CHUNK_PATH = Path(__file__).parent.parent / "preprocess"
    for pdf in CHUNK_PATH.glob("*.pdf"):
        chunks = chunk_pdf(str(pdf))
        for idx, c in enumerate(chunks):
            out = Path(CHUNKS_DIR) / f"{pdf.stem}_chunk{idx}.txt"
            out.write_text(c)
    print(f"Generated chunks in {CHUNKS_DIR}")
