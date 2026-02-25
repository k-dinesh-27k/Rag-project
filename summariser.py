import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ==============================
# CONFIG
# ==============================

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

INPUT_FILE = "daa.txt"                 # Your full book text
CHUNKS_FILE = "chunks.txt"               # Intermediate
OUTPUT_SUMMARY_FILE = "chunk_summaries.txt"
FINAL_SUMMARY_FILE = "final_summary.txt"

CHUNK_SIZE = 10000
OVERLAP = 500

MAX_NEW_TOKENS = 60
MIN_NEW_TOKENS = 20

# ==============================
# LOAD MODEL
# ==============================

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to("cpu")
print("Model loaded.\n")

# ==============================
# READ FULL TEXT
# ==============================

def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# ==============================
# CHUNKING (same logic as before)
# ==============================

def split_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks

def save_chunks(chunks):
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write("=====CHUNK=====\n")
            f.write(chunk.strip() + "\n")

# ==============================
# SUMMARIZATION
# ==============================

def summarize_chunk(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=MAX_NEW_TOKENS,
        min_new_tokens=MIN_NEW_TOKENS,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ==============================
# MAIN
# ==============================

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

    # Save intermediate summaries
    with open(OUTPUT_SUMMARY_FILE, "w", encoding="utf-8") as f:
        for s in summaries:
            f.write(s.strip() + "\n")

    # Combine final summary
    final_summary = " ".join(summaries)

    with open(FINAL_SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(final_summary)

    print("\nDone.")
    print("Chunks saved →", CHUNKS_FILE)
    print("Chunk summaries →", OUTPUT_SUMMARY_FILE)
    print("Final summary →", FINAL_SUMMARY_FILE)

if __name__ == "__main__":
    main()