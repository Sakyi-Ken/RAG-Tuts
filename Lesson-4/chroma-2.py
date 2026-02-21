import json
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions


def retrieve_top_chunks(query, collection, top_k=2):
    """
    Retrieves the top_k most relevant to the given query from 'collection.
    Returns a list of dictionaries containing 'chunk' text, 'doc_id', and 'distance'.
    """
    # Search for top_k results matching the user's query
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    retrieved_chunks = []

    # Safeguard in case no results are found.
    if not results['documents'] or not results['documents'][0]:
        return retrieved_chunks
    
    # Gather each retrieved chunk, along with its distance score
    for i in range(len(results['documents'][0])):
        retrieved_chunks.append({
            "chunk": results['documents'][0][i],
            "doc_id": results['ids'][0][i],
            "distance": results['distances'][0][i]
        })
    return retrieved_chunks

# load corpus data from JSON file
with open("../data/corpus.json", 'r') as f:
    corpus_data = json.load(f)

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
client = Client(Settings())
collection = client.get_or_create_collection("rag_collection", embedding_function=embed_func)

# Batch add documents from the corpus data
documents = [doc['content'] for doc in corpus_data]
ids = [f"chunk_{doc['id']}_0" for doc in corpus_data]
collection.add(documents=documents, ids=ids)