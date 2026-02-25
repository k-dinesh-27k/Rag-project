import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



MODEL_NAME = "sshleifer/distilbart-cnn-12-6" #Using a small transformer model for chunk wise summarisation

INPUT_FILE = "daa.txt" # File which contains whole book data
CHUNKS_FILE = "chunks.txt" #Saving each chunk in a text file              
OUTPUT_SUMMARY_FILE = "chunk_summaries.txt" #Saving all chunk wise summary in a file
FINAL_SUMMARY_FILE = "final_summary.txt" #Creating a whole summary file

CHUNK_SIZE = 10000 #Keeping chink larger to minimise time and iterations and keeping context memory high
OVERLAP = 500

MAX_NEW_TOKENS = 60
MIN_NEW_TOKENS = 20


print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) #Importing the tokeniser required for model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME) #Importing the model
model.to("cpu")
print("Model loaded.\n")


#Reading the input file
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


#Chunking the data , using the thresholds intialised at the start
def split_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap #Overlapping the data to not lose context

    return chunks
#Funtion to save each chunk in a text file
def save_chunks(chunks):
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write("=====CHUNK=====\n")
            f.write(chunk.strip() + "\n")
#Function to summarise each chunk 
def summarize_chunk(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True, # Using turincation to summarise the chunk
        max_length=1024
    )#Tokenising the input 

    summary_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=MAX_NEW_TOKENS,#Limiting the tokens , so that model will sumarise the data without mentioning
        min_new_tokens=MIN_NEW_TOKENS,
        num_beams=4,#It is used to tell the model how many sequence to keep track of.
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True) #Returning the decoded response. Converts encoded data into actual english text


def main():

    print("Reading input file...")
    full_text = load_text(INPUT_FILE)

    print("Splitting into chunks...")
    chunks = split_text(full_text)
    print(f"Created {len(chunks)} chunks.\n")

    save_chunks(chunks)

    summaries = []

    for i, chunk in enumerate(chunks):
        print(f"Summarizing chunk {i+1}/{len(chunks)}...")
        try:
            summary = summarize_chunk(chunk)
            summaries.append(summary)
        except Exception as e:
            print("Error:", e)
#Storing chunk wise summary in text file
    with open(OUTPUT_SUMMARY_FILE, "w", encoding="utf-8") as f:
        for s in summaries:
            f.write(s.strip() + "\n")

    final_summary = " ".join(summaries)
#Storing the final summary in the text file
    with open(FINAL_SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(final_summary)

    print("\nDone.")
    print("Chunks saved →", CHUNKS_FILE)
    print("Chunk summaries →", OUTPUT_SUMMARY_FILE)
    print("Final summary →", FINAL_SUMMARY_FILE)

if __name__ == "__main__":
    main()