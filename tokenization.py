import nltk
from nltk.tokenize import word_tokenize

# Download tokenizer data once
nltk.download("punkt")
nltk.download("punkt_tab")

documents = [
    "Natural Language Processing is transforming Artificial Intelligence.",
    "Machine learning models require clean and normalized text data.",
    "Tokenization, stemming, and lemmatization are important NLP preprocessing steps."
]

for index, document in enumerate(documents, start=1):
    tokens = word_tokenize(document)

    # Keep only words and remove punctuation
    word_tokens = [token.lower() for token in tokens if token.isalpha()]

    print(f"Document {index}:")
    print(word_tokens)
    print("Total tokens:", len(word_tokens))
    print()