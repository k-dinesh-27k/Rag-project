import chromadb
import requests
import re


def ollama_embed(text, model="nomic-embed-text"):  #Using local ollama model to embed data and store it in vector DB
    response = requests.post(
        "http://host.docker.internal:11434/api/embeddings",
        json={"model": model, "prompt": text}
    )
    return response.json()["embedding"]
#Initialising a vector database with local storage.
client = chromadb.PersistentClient(path="ChromaDB")

collection = client.get_or_create_collection(name="Rag_book")


print("Loading text file...")
#Reading the text of the cleaned book data
with open("daa.txt", "r", encoding="utf-8") as f:
    text = f.read()
#splitting the data into pages using regex
pages = re.split(r'\[Page[^\]]*\]', text)
#Chunking the data as each page as a chunk , as the model used taskes more time for each chunk , and page wise chunking keeps context of the chunk within a chunk
chunks = [p.strip() for p in pages if p.strip()]

print("Total chunks:", len(chunks))
print("Text length:", len(text))
print("Storing into Chroma...")
#Enumerating the chunks to keep track of ID of each chunk in RAG 
for i, j in enumerate(chunks):
    print(f"Embedding {i+1}/{len(chunks)}", end="\r")

    p = ollama_embed(j) # Using embedding function to embed the data

    collection.add(
        ids=[str(i)], #Using serial numbers as IDs
        documents=[j], #The raw chunk is also stored to retrieve when needed
        embeddings=[p] #Embedding of each chunk is stored 
    )
print("Data stored.")
