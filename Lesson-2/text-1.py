KNOWLEDGE_BASE = [
    "Retrieval-Augmented Generation (RAG) enhances language models by integrating relevant external documents into the generation process.",
    "RAG systems retrieve information from large databases to provide contextual answers beyond what is stored in the model.",
    "By merging retrieved text with generative models, RAG overcomes the limitations of static training data.",
    "Media companies combine external data feeds with digital editing tools to optimize broadcast schedules.",
    "Financial institutions analyze market data and use automated report generation to guide investment decisions.",
    "Healthcare analytics platforms integrate patient records with predictive models to generate personalized care plans.",
    "Bananas are popular fruits that are rich in essential nutrients such as potassium and vitamin C."
]

def build_vocabs(docs):
    unique_words = set()
    for doc in docs:
        for word in doc.lower().split():
            clean_word = word.strip(".,!?")
            if clean_word:
                unique_words.add(clean_word)
    return {word: idx for idx, word in enumerate(sorted(unique_words))}

VOCAB = build_vocabs(KNOWLEDGE_BASE)


import numpy as np

def bow_vectorize(text , vocab):
    vector = np.zeros(len(vocab), dtype=int)
    for word in text.lower().split():
        clean_word = word.strip(".,!?")
        if clean_word in vocab:
            index = vocab[clean_word]
            vector[index] += 1
    return vector

def bow_search(query, docs):
    """
    Rank documents by lexical overlap using the BOW technique.
    The dot product between the query vector and each document
    indicates how many words they share.
    """
    query_vec = bow_vectorize(query, VOCAB)
    scores = []
    for i, doc in enumerate(docs):
        doc_vec = bow_vectorize(doc, VOCAB)
        score = np.dot(query_vec, doc_vec)
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

def preprocess_string(text):
    words = text.split()
    cleaned_tokens = [word.lower().strip(".,!?") for word in words]
    return cleaned_tokens

if __name__ == "__main__":
    query = "How does a system combine external data with language generation to improve responses?"
    print(f"Query: {query}")

    # BOW-based search
    bow_results = bow_search(query, KNOWLEDGE_BASE)
    print("BOW Search Results:")
    for idx, score in bow_results:
        print(f"  Doc {idx} | Score: {score} | Text: {KNOWLEDGE_BASE[idx]}")