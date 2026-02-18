import os
import json

def chunk_text(text, chunk_size=10, keywords=None):
    # TODO: Add keywords parameter to function
    """
    Splits the given text into chunks of size 'chunk_size' and detects keywords.
    Returns a list of dictionaries containing chunk text and found keywords.
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i+chunk_size]
        chunk_text = " ".join(chunk_words)
        
        # TODO: Create a set to store found keywords in this chunk
        found_keywords = set()
        
        # TODO: Check each word in the chunk against keywords
        # If a word is in keywords, add it to found keywords
        if keywords:
            for word in chunk_words:
                if word in keywords:
                    found_keywords.add(word)
        
        chunks.append({
            "text": chunk_text,
            "keywords": list(found_keywords)
        })
    
    return chunks


def load_and_chunk_dataset(file_path, chunk_size=30, keywords=None):

    """
    Loads a dataset from JSON 'file_path', splits each document into smaller chunks,
    and detects keywords in each chunk.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    
    all_chunks = []
    for doc_id, doc in enumerate(data):
        doc_text = doc["content"]
        doc_category = doc.get("category", "general")
        doc_chunks = chunk_text(doc_text, chunk_size, keywords=keywords)
        
        for chunk_id, chunk_data in enumerate(doc_chunks):
            chunk_data.update({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "category": doc_category,
            })
            all_chunks.append(chunk_data)
    
    return all_chunks


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    dataset_file = os.path.join(current_dir, "data", "corpus.json")
    
    # Define keywords to track
    keywords_to_track = ["Artificial Intelligence", "technology", "cinema"]
    
    # Load and chunk the dataset with keyword detection
    chunked_docs = load_and_chunk_dataset(
        dataset_file, 
        chunk_size=30, 
        keywords=keywords_to_track
    )
    
    print("Loaded and chunked", len(chunked_docs), "chunks from dataset.")
    for chunk in chunked_docs:
        print("\nChunk:")
        print("Text:", chunk["text"])
        print("Keywords found:", chunk["keywords"])