import numpy as np
from numpy.linalg import norm

def cosine_similarity(vec_a, vec_b):
    """
    Compute cosine similarity between two vectors.
    Range: -1 (Opposite directions) to 1 (Same direction), with 0 indicating orthogonality.
    """
    if norm(vec_a) == 0 or norm(vec_b) == 0:
        return 0.0
    return np.dot(vec_a, vec_b) / (norm(vec_a) * norm(vec_b)) 

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

sentences = [
    "RAG stands for Retrieval Augmented Generation.",
    "A Large Language Model is a Generative AI model for text generation.",
    "RAG enhance text generation of LLMs by incorporating external data",
    "Bananas are yellow fruits.",
    "Apples are good for your health.",
    "What's monkey's favorite food?"
]

# embeddings = model.encode(sentences)
# print(embeddings.shape)  # e.g., (6, 384), depending on the model
# print(embeddings[0])     # A sample embedding for the first sentence

# Compare the sentences to themselves.

# for i, sent_i in enumerate(sentences):
#     for j, sent_j in enumerate(sentences[i+1:], start=i+1):
#         sim_score = cosine_similarity(embeddings[i], embeddings[j])
#         print(f"Similarity('{sent_i}', '{sent_j}') = {sim_score:.4f}")

# Compare all other sentences to the first sentence

#map = {}
# map = []
# one_sentence = sentences[0]
# if one_sentence in sentences:
#     for j, sent_j in enumerate(sentences[1:], start=1):
#         sim_score = cosine_similarity(embeddings[0], embeddings[j])
#         map[(one_sentence, sent_j)] = sim_score
        # map.append((one_sentence _ sent_j)) = sin_score
# map.sort(key=lambda x: x[1], reverse=True)
# ranked = sorted(map.items(), key=lambda x: x[1], reverse=True)
# print("Ranked sentence pairs by similarity:")
# for pair, score in ranked:
#     print(f"Similarity('{pair[0]}', '{pair[1]}') = {score:.4f}")

# Compare the query to the sentences in the docs.

def embedding_search(query, docs, model):
    """
    Rank documents by comparing how semantically close they are
    to the query in the embedding space using cosine similarity.
    """
    query_emb = model.encode([query])[0]
    docs_emb = model.encode(docs)

    print(query_emb, "Query Embeddings")

    scores = []
    for i, emb in enumerate(docs_emb):
        score = cosine_similarity(query_emb, emb)
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

query = "How does a system combine external data with language generation to improve responses?"

KNOWLEDGE_BASE = [
    "Retrieval-Augmented Generation (RAG) enhances language models by integrating relevant external documents into the generation process.",
    "RAG systems retrieve information from large databases to provide contextual answers beyond what is stored in the model.",
    "By merging retrieved text with generative models, RAG overcomes the limitations of static training data.",
    "Media companies combine external data feeds with digital editing tools to optimize broadcast schedules.",
    "Financial institutions analyze market data and use automated report generation to guide investment decisions.",
    "Healthcare analytics platforms integrate patient records with predictive models to generate personalized care plans.",
    "Bananas are popular fruits that are rich in essential nutrients such as potassium and vitamin C."
]

scores = embedding_search(query, KNOWLEDGE_BASE, model)
print(scores)