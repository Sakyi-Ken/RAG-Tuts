def build_vocabs(docs):
    unique_words = set()
    for doc in docs:
        for word in doc.lower().split():
            clean_word = word.strip(".,!?")
            if clean_word:
                unique_words.add(clean_word)
    return {word: idx for idx, word in enumerate(sorted(unique_words))}


import numpy as np

def bow_vectorize(text , vocab):
    vector = np.zeros(len(vocab), dtype=int)
    for word in text.lower().split():
        clean_word = word.strip(".,!?")
        if clean_word in vocab:
            index = vocab[clean_word]
            vector[index] += 1
    return vector


def preprocess_string(text):
    words = text.split()
    cleaned_tokens = [word.lower().strip(".,!?") for word in words]
    return cleaned_tokens