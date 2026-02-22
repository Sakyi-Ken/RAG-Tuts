import json
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions
#from scripts.llm import get_llm_response


def retrieve_top_chunks(query, collection, category=None, top_k=3):
    """
    Retrieves the top_k chunks most relevant to the given query from 'collection',
    optionally filtered by category. Returns a list of retrieved chunks, each
    containing 'chunk' text, 'doc_id', and 'distance'.
    """
    # TODO: Create a where dictionary to filter by category if one is provided
    where = {}
    if category:
        where = {"category": category}

    # TODO: Perform the query with metadata filtering using collection.query()
    # Include the where parameter in the query
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where
    )

    retrieved_chunks = []

    # Safeguard against empty results
    if not results['documents'] or not results['documents'][0]:
        return retrieved_chunks

    # TODO: Process query results and append each chunk's information to retrieved_chunks
    for i in range(len(results['documents'][0])):
        retrieved_chunks.append({
            'chunk': results['documents'][0][i],
            'doc_id': results['ids'][0][i],
            'distance': results['distances'][0][i]
        })

    return retrieved_chunks


def build_prompt(query, retrieved_chunks):
    """
    Constructs an LLM prompt by combining multiple retrieved chunks into a
    single context block, ensuring the model can provide context-based answers.
    """
    prompt = f"Question: {query}\nAnswer using only the following context:\n"
    for rc in retrieved_chunks:
        prompt += f"- {rc['chunk']}\n"
    prompt += "Answer:"
    return prompt


if __name__ == "__main__":
    # Load corpus data from JSON file
    with open('data/corpus.json', 'r') as f:
        corpus_data = json.load(f)

    # Prepare documents, ids, and metadatas
    documents = [doc['content'] for doc in corpus_data]
    ids = [f"chunk_{doc['id']}_0" for doc in corpus_data]
    metadatas = [{"category": doc["category"]} for doc in corpus_data]

    # Create or retrieve the vector database collection
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
    client = Client(Settings())
    collection = client.get_or_create_collection(
        "rag_collection",
        embedding_function=embed_func
    )

    # Add documents with metadata to the collection
    collection.add(documents=documents, ids=ids, metadatas=metadatas)

    # TODO: Define a query and category to test the retrieval function
    user_query = "What are some recent technological breakthroughs?"
    user_category = "technology"

    # TODO: Retrieve chunks matching the query and category
    retrieved = retrieve_top_chunks(user_query, collection, category=user_category, top_k=3)

    # TODO: Implement logic to handle empty results, build prompt, and get LLM response
    # Print appropriate messages or the final prompt and answer
    if not retrieved:
        print("No relevant chunks found for the given query and category.")
    else:
        final_prompt = build_prompt(user_query, retrieved)
        #answer = get_llm_response(final_prompt)
        print("Prompt:\n")
        print(final_prompt)
        print("\nLLM Response:\n")
        #print(answer)