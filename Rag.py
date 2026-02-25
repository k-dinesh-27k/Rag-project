from pypdf import PdfReader
import chromadb
import requests
import re


def ollama_embed(text, model="nomic-embed-text"):  # or your model name
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": model, "prompt": text}
    )
    return response.json()["embedding"]

# ✅ NEW API — this actually creates the folder
client = chromadb.PersistentClient(path="ChromaDB")

collection = client.get_or_create_collection(name="Rag_book")


print("Loading text file...")

with open("daa.txt", "r", encoding="utf-8") as f:
    text = f.read()

pages = re.split(r'\[Page[^\]]*\]', text)
chunks = [p.strip() for p in pages if p.strip()]

print("Total chunks:", len(chunks))
print("Text length:", len(text))
print("Storing into Chroma...")

for i, j in enumerate(chunks):
    print(f"Embedding {i+1}/{len(chunks)}", end="\r")

    p = ollama_embed(j)

    collection.add(
        ids=[str(i)],
        documents=[j],
        embeddings=[p]
    )

print("\n✅ Done. Data stored.")

print("✅ Done. Data stored.")