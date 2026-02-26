# Summarisation and RAG Implementation

## Summarisation:(Summariser.py)

Task: To summarize the given book named cryptography  

Steps:

1) Extracted whole text from the book using pymupdf library in python  
2) Cleaned the whole book , by removing unwanted newline chars , removed the page numbers in the data and removed appendixes and reference hyperlinks in data  
3) As the size of the data is large, The data was chunked with 10000 chars and , 500 chars as overlapping between chunks  
4) Used a small transformer model (distilbart) to summarise chunks to keep important details of each chunk  
5) Merged all the important details of each chunk into a large chunk  
6) Used Ollama's mistral LLM to refine the summary (Final_summary.py)  

---

## RAG:(Rag.py)

Task: To implement RAG system to the given book named cryptography  

Steps:

1) Used chromaDB for vector database  
2) Used nomic embedding model from Ollama  
3) Chunked the data by pages to keep context of the chunk within a chunk  
4) Embedded each chunk using embedding model  
5) Stored the chunk in ChromaDB  

---

## Chatbot with RAG:(query.py)

Task: To implement a chatbot to ask queries in the given book  

Steps:

1) Used same chromaDB vector database used in rag.py to retrieve the required context data  
2) Used Ollama's mistral(7B parameter) LLM to answer queries asked with given context  
3) Used same nomic embedding model , to retrieve the relevant data stored in the vectorDB  
4) Most 5 related chunks are retrieved from the vector Database  
5) The relevant chunks are given to LLM in a prompt to answer the query based on given context  
6) Models response is displayed  

---

## Summaries Folder:

This folder consist of , chunks of the data , summaries of each chunk ,a final chunk from transformer model and a refined LLMs summary of the whole book  

---

## Daa.txt :

Contains the whole cleaned data from the book  
