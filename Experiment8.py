
import nltk
import numpy as np
from nltk.corpus import wordnet as wn
from gensim.downloader import load


nltk.download("wordnet")
nltk.download("omw-1.4")


model = load("glove-wiki-gigaword-50")



def wordnet_similarity(word1, word2):
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)

    max_similarity = 0

    for s1 in synsets1:
        for s2 in synsets2:
            similarity = s1.path_similarity(s2)

            if similarity is not None:
                max_similarity = max(max_similarity, similarity)

    return max_similarity




def word_similarity(word1, word2):
    if word1 in model and word2 in model:
        return model.similarity(word1, word2)
    return 0



def text_vector(text):
    words = text.lower().split()

    vectors = [
        model[word]
        for word in words
        if word in model
    ]

    if not vectors:
        return np.zeros(model.vector_size)

    return np.mean(vectors, axis=0)


def text_similarity(text1, text2):
    vector1 = text_vector(text1)
    vector2 = text_vector(text2)

    denominator = (
        np.linalg.norm(vector1) *
        np.linalg.norm(vector2)
    )

    if denominator == 0:
        return 0

    return np.dot(vector1, vector2) / denominator




word1 = "car"
word2 = "automobile"

print("WORD SIMILARITY")
print("-------------------------")

print("Words:", word1, "and", word2)

print(
    "WordNet Similarity:",
    round(wordnet_similarity(word1, word2), 4)
)

print(
    "Embedding Similarity:",
    round(word_similarity(word1, word2), 4)
)


phrase1 = "I love eating food"
phrase2 = "I enjoy having meals"

print("\nPHRASE SIMILARITY")
print("-------------------------")

print("Phrase 1:", phrase1)
print("Phrase 2:", phrase2)

print(
    "Similarity:",
    round(text_similarity(phrase1, phrase2), 4)
)


document1 = """
Artificial intelligence is transforming modern technology.
Machine learning is widely used in intelligent applications.
"""

document2 = """
AI and machine learning are changing the technology industry.
Intelligent systems are becoming increasingly common.
"""

print("\nDOCUMENT SIMILARITY")
print("-------------------------")

print("Document 1:", document1)
print("Document 2:", document2)

print(
    "Similarity:",
    round(text_similarity(document1, document2), 4)
)