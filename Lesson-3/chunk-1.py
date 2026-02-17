data = [
  {
    "id": 1,
    "content": "Hello world! This is a sample document used for testing chunk_text function."
  },
  {
    "id": 2,
    "content": "Another sample document. This is used for verifying the chunking of text in multiple documents. It includes additional sentences to provide a more comprehensive test case. By having a longer document, we can better assess how the chunking function performs when dealing with more extensive content."
  }
]

def chunk_text(text, chunk_size=10):
    """
    Splits the given text into smaller chunks, each containing
    up to 'chunk_size' words. Returns a list of these chunk strings.
    """
    words = text.split() # tokenize by splitting on whitespace.
    # Construct chunks by stepping through the words list in increments of chunk_size.
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def load_and_chunk_dataset(data, chunk_size=10):
    """
    Iterates over a structured dataset of documents, splits each into chunks,
    and associates metadata (doc_id and chunk_id) with every place.
    """
    all_chunks = []
    for doc in data:
        doc_id = doc["id"]
        doc_text = doc["content"]

        # Create a smaller text segments from the original document
        doc_chunks = chunk_text(doc_text, chunk_size)

        # Label each chunk with its source identifier
        for chunk_id, chunk_str in enumerate(doc_chunks):
            all_chunks.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": chunk_str
            })
        return all_chunks
    
chunked_docs = load_and_chunk_dataset(data, chunk_size=10)
print(f"Loaded and chunked {len(chunked_docs)} chunks from the dataset.")
for doc in chunked_docs:
    print(doc)