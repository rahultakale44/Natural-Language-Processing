import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


# Given raw text documents
documents = [
    "Natural Language Processing is transforming Artificial Intelligence.",
    "Machine learning models require clean and normalized text data.",
    "Tokenization, stemming, and lemmatization are important NLP preprocessing steps."
]


# Initialize tools
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    print("\nOriginal Text:")
    print(text)

    # 1. Normalization
    normalized_text = text.lower()

    # Remove punctuation and special characters
    normalized_text = re.sub(r"[^a-zA-Z\s]", "", normalized_text)

    print("\nNormalized Text:")
    print(normalized_text)

    # 2. Tokenization
    tokens = word_tokenize(normalized_text)

    print("\nTokens:")
    print(tokens)

    # 3. Stop-word Removal
    filtered_tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    print("\nAfter Stop-word Removal:")
    print(filtered_tokens)

    # 4. Stemming
    stemmed_words = [
        stemmer.stem(word)
        for word in filtered_tokens
    ]

    print("\nStemmed Words:")
    print(stemmed_words)

    # 5. Lemmatization
    lemmatized_words = [
        lemmatizer.lemmatize(word)
        for word in filtered_tokens
    ]

    print("\nLemmatized Words:")
    print(lemmatized_words)

    return lemmatized_words


print("=" * 70)
print("NLP TEXT PREPROCESSING PIPELINE")
print("=" * 70)


processed_documents = []

for index, document in enumerate(documents, start=1):

    print("\n" + "=" * 70)
    print(f"DOCUMENT {index}")
    print("=" * 70)

    processed_document = preprocess_text(document)

    processed_documents.append(processed_document)


print("\n" + "=" * 70)
print("FINAL PREPROCESSED DOCUMENTS")
print("=" * 70)

for index, document in enumerate(processed_documents, start=1):
    print(f"Document {index}: {document}")