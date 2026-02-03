#1. Indexing: Organising Documents
KNOWLEDGE_BASE = {
    "doc1": {
        "title": "Project Chimera Overview",
        "content": (
            "Project Chimera is a research initiative focused on developing "
            "novel bio-integrated interfaces. It aims to merge biological "
            "systems with advanced computing technologies."
        )
    },
    "doc2": {
        "title": "Chimera's Neural Interface",
        "content": (
            "The core component of Project Chimera is a neural interface "
            "that allows for bidirectional communication between the brain "
            "and external devices. This interface uses biocompatible "
            "nanomaterials."
        )
    },
    "doc3": {
        "title": "Applications of Chimera",
        "content": (
            "Potential applications of Project Chimera include advanced "
            "prosthetics, treatment of neurological disorders, and enhanced "
            "human-computer interaction. Ethical considerations are paramount."
        )
    }
}

#2. Retrieval: Locating Relevant Information
def rag_retrieval(query, documents):
    query_words = set(query.lower().split())
    best_doc_id = None
    best_overlap = 0
    
    for doc_id, doc in documents.items():
        # Compare the query words with the document's content words
        doc_words = set(doc["content"].lower().split() + doc["title"].lower().split())
        overlap = len(query_words.intersection(doc_words))
        
        if overlap > best_overlap:
            best_overlap = overlap
            best_doc_id = doc_id
    
    # Return the best document, or None if nothing matched
    return documents.get(best_doc_id)

#3. Query Augmentation: Creating Context-Rich Prompts
def rag_generation(query, document):
    if document:
        snippet = f"{document['title']}: {document['content']}"
        prompt = f"Using the following information: '{snippet}', answer: {query}"
    else:
        prompt = f"No relevant information found. Answer directly: {query}"
    return get_llm_response(prompt)

def get_llm_response(prompt):
    """
    This function interfaces with a language model to generate a response based on the provided prompt.
    
    Parameters:
    - prompt (str): A string containing the question or task for the language model, potentially augmented with additional context.
    
    Returns:
    - response (str): The generated text from the language model, which aims to answer the question or fulfill the task described in the prompt.
    """
    pass

def naive_generation(query):
    # This approach ignores the knowledge base
    prompt = f"Answer directly the following query: {query}"
    return get_llm_response(prompt)

# Example Usage
if __name__ == "__main__":
    query = "What are the applications of Project Chimera?"
    print("Naive approach:", naive_generation(query))
    retrieved_docs = rag_retrieval(query, KNOWLEDGE_BASE)
    print("RAG approach:", rag_generation(query, retrieved_docs))
    