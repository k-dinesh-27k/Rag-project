import requests

def ollama_response(data):
    prompt = f"You are given a summary of a whole book and it was summarised with small models where summarisation is still too long.\n I want you to summarise the given text even further to make it meaningful and try to keep the main idea of the book as much as possible , data :{data}"

    response = requests.post(
       "http://host.docker.internal:11434/api/generate",
        json={"model": "mistral", "prompt": prompt, "stream" : False}
    )
    return response.json()['response']

with open("final_summary.txt",'r') as f:
    data = f.read()
summary = ollama_response(data)
print(summary)
with open("Refined_whole_summary.txt",'w',encoding="utf-8") as f:
    f.write(summary)
