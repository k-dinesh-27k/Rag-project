import chromadb
import requests
import sys


PERSIST_DIR = "ChromaDB" #Using the vector data base created using Rag.py to retrieve relevent information

def ollama_embed(text, model="nomic-embed-text"): #Used same embedding model which was used to implement RAG
    response = requests.post(
        "http://host.docker.internal:11434/api/embeddings",
        json={"model": model, "prompt": text}
    )
    return response.json()["embedding"]

def ollama_response(text, model="mistral"): #Using Mistral LLM model to refine the response, The context required for query is given in prompt using RAG implementation
    response = requests.post(
       "http://host.docker.internal:11434/api/generate",
        json={"model": model, "prompt": text, "stream" : False}
    )
    return response.json()['response']
print("Connecting to Chroma DB...")
client = chromadb.PersistentClient(path=PERSIST_DIR)# Connecting with same DB used in RAG.py

collection = client.get_collection(name="Rag_book")

#Function to retrieve relevent context data from Vector database
def retrieve_context(query, k=5):
    query_embedding = ollama_embed(query) # Embedding the query 

    results = collection.query(
        query_embeddings=[query_embedding],   #Retrieveng most 5 relevent chunks from whole book to give context to the LLM model about the query asked
        n_results=k #Chose 5 as the number of chunks to be retrieved
    )

    return results["documents"][0]



    



if __name__ == "__main__":
  

     
    while True:
       
        if len(sys.argv) > 1:
            query = " ".join(sys.argv[1:])
        else:
            query = input("Ask a question: ")

        if query.lower() == "exit":
            break

        contexts = retrieve_context(query)

        print("\nTop Retrieved Chunks:\n")
        for i, c in enumerate(contexts):
            print(f"[Chunk {i}] {c[:200]}...\n")
        joined = "\n\n".join(contexts)#Joining all chunks into one
        prompt = f"""
        You must answer ONLY using the provided context.
        If the answer is not in the context, say "Not found".

        Context:
        {joined}

        Question:
        {query}
        """

        
        print("Model response: ",ollama_response(prompt))