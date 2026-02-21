from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions

def build_chroma_collection(chunks):
    # Use a Sentence Transformer model for embeddings
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    embed_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

    # Create a ChromaDB client with default settings
    client = Client(Settings())

    # Either get an exiting collection or create a new one 
    collection = client.get_or_create_collection(
        name="rag_collection",
        embedding_function=embed_func    
    )

    # Prepare the data: texts, IDs, and metadata
    texts = [c["content"] for c in chunks]
    ids = [f"chunk_{c['doc_id']}_{c['chunk_id']}" for c in chunks]
    metadatas = [
        {"doc_id": chunk["doc_id"],
         "chunk_id": chunk["chunk_id"],
         "category": chunk["category"]}
         for chunk in chunks
    ]

    # Add the documents (chunks) to the collection
    collection.add(documents=texts, metadatas=metadatas, ids=ids)
    return collection


# Example chunks to showcase adding them to a new collection
example_chunks = [
    {"doc_id": 0, "chunk_id": 0, "category": "ai", "content": "RAG stands for Retrieval-Augmented Generation."},
    {"doc_id": 0, "chunk_id": 1, "category": "ai", "content": "A crucial component of a RAG pipeline is the Vector Database."},
    {"doc_id": 1, "chunk_id": 0, "category": "finance", "content": "Accurate data is essential in finance."},
]
collection = build_chroma_collection(example_chunks)

# Prepare a new chunk to add
new_document = {
    "doc_id": 2,
    "chunk_id": 0,
    "category": "food",
    "content": "Bananas are yellow fruits rich in potassium."
}

# Construct a unique ID for the new document
# Format: "chunk_{doc_id}_{chunk_id}" (e.g., "chunk_2_0")
doc_id = f"chunk_{new_document['doc_id']}_{new_document['chunk_id']}"

# Add the new chunk to the existing collection
collection.add(
    documents=[new_document["content"]], # The text content to be embedded
    metadatas=[{                         # Metadata for filtering and context
        "doc_id": new_document["doc_id"],
        "chunk_id": new_document["chunk_id"],
        "category": new_document["category"]
    }],
    ids=[doc_id]  # Unique identifier for this chunk
)

# If needed, remove the chunk by its unique ID
# For example, if the information about bananas becomes outdated
collection.delete(ids=[doc_id])  # Using the same ID: "chunk_2_0"

def delete_documents_with_keyword(collection, keyword):
    """
    Deletes all documents from the given ChromaDB 'collection' whose text contains 'keyword'.
    """
    # TODO: Get all documents and their IDs from the collection
    all_docs = collection.get()
    documents = all_docs["documents"]
    ids = all_docs["ids"]

    # TODO: Create a list to store IDs of documents containing the keyword
    ids_to_delete = []

    # TODO: Iterate through documents and their IDs, adding matching document IDs to the list
    for i, doc in enumerate(documents):
        if keyword.lower() in doc.lower():
            ids_to_delete.append(ids[i])

    # TODO: If there are documents to delete, remove them from the collection
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)

    # Alternative
    # for doc_id, doc in zip(ids, documents):
    #     if keyword.lower() in doc.lower():
    #         ids_to_delete.append(doc_id)