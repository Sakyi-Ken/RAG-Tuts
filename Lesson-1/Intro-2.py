# from scripts.llm import get_llm_response

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

def naive_generation(query):
    prompt = f"Answer directly the following query: {query}"
    return prompt #get_llm_response(prompt)


def rag_retrieval(query, documents):
    query_words = set(query.lower().split())
    docs = []
    for doc_id, doc in documents.items():
        doc_words = set(doc["content"].lower().split())
        overlap = len(query_words.intersection(doc_words))
        if overlap > 0:
            docs.append(documents.get(doc_id))
    # TODO: Modify this function to return ALL relevant documents
    # and not just the one with the highest overlap.
    return docs


def rag_generation(query, documents):
    # TODO: Modify rag_generation to handle a list of documents.
    snippets = []
    if documents:
        for doc in documents:
            snippet = f"{doc['title']}: {doc['content']}"
            snippets.append(snippet)
        prompt = f"Using the following information: '{snippets}', answer: {query}"
    else:
        prompt = f"No relevant information found. Answer directly: {query}"
    return prompt #get_llm_response(prompt)


if __name__ == "__main__":
    query = "What are the applications of Project Chimera?"
    print("Naive approach:", naive_generation(query))
    retrieved_docs = rag_retrieval(query, KNOWLEDGE_BASE)
    print("RAG approach:", rag_generation(query, retrieved_docs))