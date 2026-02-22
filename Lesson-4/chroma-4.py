import json
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions


def retrieve_top_chunks(query, collection, category=None, top_k=3, distance_threshold=1.0):
    """
    Retrieves the top_k chunks most relevant to the given query from 'collection',
    optionally filtered by category, and only includes those whose distance is
    below the specified distance_threshold. Returns a list of retrieved chunks,
    each containing 'chunk', 'doc_id', and 'distance'.
    """
    where = {"category": category} if category is not None else None
    if where is not None:
        where["distance"] = {"$lte": distance_threshold}  # Filter by distance threshold

    results = collection.query(
        query_texts=[query],
        where=where,
        n_results=top_k
    )


    retrieved_chunks = []
    if not results["documents"] or not results["documents"][0]:
        return retrieved_chunks

    # TODO: Process the results and append chunks that meet the distance threshold
    # For each chunk in results, check if its distance is below distance_threshold
    # If it qualifies, add it to retrieved_chunks with chunk text, doc_id, and distance
    for i in range(len(results["documents"][0])):
        retrieved_chunks.append({
            "chunk": results["documents"][0][i],
            "doc_id": results["ids"][0][i],
            "distance": results["distances"][0][i]
        })

    return retrieved_chunks


def build_prompt(query, retrieved_chunks):
    """
    Constructs a prompt by combining the query and retrieved chunks into a
    context block, guiding the LLM to provide a context-based answer.
    """
    prompt = f"Question: {query}\nAnswer using only the following context:\n"
    for rc in retrieved_chunks:
        prompt += f"- {rc['chunk']}\n"
    prompt += "Answer:"
    return prompt


if __name__ == "__main__":
    # Load corpus data from JSON file
    with open("data/corpus.json", "r") as f:
        corpus_data = json.load(f)

    # Prepare documents, ids, and metadatas
    documents = [doc["content"] for doc in corpus_data]
    ids = [f"chunk_{doc['id']}_0" for doc in corpus_data]
    metadatas = [{"category": doc.get("category", "")} for doc in corpus_data]

    # Create or retrieve the vector database collection
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    client = Client(Settings())
    collection = client.get_or_create_collection("rag_collection", embedding_function=embed_func)

    # Add documents with metadata to the collection
    collection.add(documents=documents, ids=ids, metadatas=metadatas)

    # Define query parameters (query string, category, and distance threshold)
    user_query = "What are the latest AI breakthroughs?"  # Example query
    user_category = "Technology"
    threshold = 1.0

    # Retrieve and filter chunks
    filtered_chunks = retrieve_top_chunks(
        query=user_query,
        collection=collection,
        category=user_category,
        top_k=5,
        distance_threshold=threshold
    )

    # TODO: Handle the filtered chunks:
    # - If no chunks found, print a user-friendly message
    # - Otherwise, build the prompt and get LLM response
    
    if not filtered_chunks:
        print("No relevant chunks found for both the query and category, kindly try again")
    else:
        final_prompt = build_prompt(user_query, filtered_chunks)
        #answer = get_llm_response(final_prompt)
        print("Prompt:\n")
        print(final_prompt)
        print("\nLLM Response:\n")
        #print(answer)