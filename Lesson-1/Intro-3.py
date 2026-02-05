# from scripts.llm import get_llm_response

KNOWLEDGE_BASE = {
    "AAPL": {
        "title": "AAPL Stock (April 2023)",
        "content": (
            "On 2023-04-13, AAPL opened at $160.50, closed at $162.30, with a high of $163.00 and a low of $159.90. "
            "Trading volume was 80 million shares. "
            "On 2023-04-14, AAPL opened at $161.10, closed at $162.80, with a high of $163.50 and a low of $160.50. "
            "Trading volume was 85 million shares."
        )
    },
    "MSFT": {
        "title": "MSFT Stock (April 2023)",
        "content": (
            "On 2023-04-13, MSFT opened at $285.00, closed at $288.50, with a high of $290.00 and a low of $283.50. "
            "Trading volume was 35 million shares. "
            "On 2023-04-14, MSFT opened at $286.00, closed at $289.00, with a high of $291.50 and a low of $284.70. "
            "Trading volume was 40 million shares."
        )
    },
    "TSLA": {
        "title": "TSLA Stock (April 2023)",
        "content": (
            "On 2023-04-13, TSLA opened at $185.00, closed at $187.00, with a high of $189.00 and a low of $184.50. "
            "Trading volume was 50 million shares. "
            "On 2023-04-14, TSLA opened at $186.00, closed at $188.50, with a high of $190.00 and a low of $185.50. "
            "Trading volume was 55 million shares."
        )
    }
}


def naive_generation(query):
    prompt = f"Answer directly the following query: {query}"
    return "LLM response placeholder" #get_llm_response(prompt)


def rag_retrieval(query, knowledge_base, k=2):    
    # TODO: Convert the query to lowercase and tokenize.
    query_words = set(query.lower().split())
    top_k_docs = []
    docs = []
    # scores = {}
    
    # TODO: Convert each document content to lowercase and tokenize.
    for doc_id, doc in knowledge_base.items():
        doc_words = set(doc["content"].lower().split() + doc["title"].lower().split())
    
    # TODO: Calculate overlap score using set intersection.
        overlap = len(query_words.intersection(doc_words))
        if overlap > 0:
            docs.append((doc_id, overlap))
            # scores[doc_id] = overlap
        print("Overlap Score:", overlap)
    
    docs.sort(key=lambda x: x[1], reverse=True)
    # ranked_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    for i in range(min(k, len(docs))):
        top_k_docs.append(knowledge_base[docs[i][0]])

    return top_k_docs

def rag_generation(query, documents):
    if documents:
        # snippets = " ".join([f"{doc['title']}: {doc['content']}" for doc in documents])
        snippet = ""
        for doc in documents:
            snippet += f"{doc['title']}: {doc['content']}\n\n"
        prompt = f"Using the following information: '{snippet}', answer: {query}"
    else:
        prompt = f"No relevant information found. Answer directly: {query}"
    return "LLM Response Placeholder" #get_llm_response(prompt)


if __name__ == "__main__":
    query = (
        "Write a short summary of the stock market performance on April 14, "
        "2023 for the following symbols: AAPL, MSFT, TSLA.\n"
        "Your summary should include:\n"
        "For each symbol:\n"
        "- The opening price\n"
        "- The closing price\n"
        "- The highest and lowest prices of the day\n"
        "- The trading volume"
    )

    # Naive approach
    print("Naive approach:\n", naive_generation(query))

    # RAG approach
    top_docs = rag_retrieval(query, KNOWLEDGE_BASE, k=2)
    print("\n\nRAG approach:\n", rag_generation(query, top_docs))