import chromadb
import requests

PERSIST_DIR = "ChromaDB"

def ollama_embed(text, model="nomic-embed-text"):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": model, "prompt": text}
    )
    return response.json()["embedding"]

def ollama_response(text, model="mistral"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": text, "stream" : False}
    )
    return response.json()['response']
print("Connecting to Chroma DB...")
client = chromadb.PersistentClient(path=PERSIST_DIR)

collection = client.get_collection(name="Rag_book")


def retrieve_context(query, k=5):
    query_embedding = ollama_embed(query)

    results = collection.query(
        query_embeddings=[query_embedding],   
        n_results=k
    )

    return results["documents"][0]


# -------- Prompt Builder --------
def build_prompt(query, contexts):
    joined = "\n\n".join(contexts)

    return f"""
You must answer ONLY using the provided context.
If the answer is not in the context, say "Not found".

Context:
{joined}

Question:
{query}
"""


# -------- Interactive Loop --------
if __name__ == "__main__":
    while True:
        query = input("\nAsk a question (or 'exit'): ")

        if query.lower() == "exit":
            break

        contexts = retrieve_context(query)

        print("\nTop Retrieved Chunks:\n")
        for i, c in enumerate(contexts):
            print(f"[Chunk {i}] {c[:200]}...\n")

        prompt = build_prompt(query, contexts)

        print("\n==== SEND THIS TO YOUR OLLAMA LLM ====\n")
        print(prompt)
        print("Model response: ",ollama_response(prompt))