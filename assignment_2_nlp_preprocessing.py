import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("stopwords")

# Input documents
documents = [
    "Natural Language Processing is transforming Artificial Intelligence.",
    "Machine learning models require clean and normalized text data.",
    "Tokenization, stemming, and lemmatization are important NLP preprocessing steps."
]

# Create required objects
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Perform all operations on every document
for index, document in enumerate(documents, start=1):

    print("\n" + "=" * 70)
    print(f"DOCUMENT {index}")
    print("=" * 70)

    print("\nOriginal Sentence:")
    print(document)

    # 1. Tokenization
    tokens = word_tokenize(document)

    # Convert to lowercase and remove punctuation
    clean_tokens = [
        token.lower()
        for token in tokens
        if token.isalpha()
    ]

    # 2. Stemming
    stemmed_tokens = [
        stemmer.stem(token)
        for token in clean_tokens
    ]

    # 3. Lemmatization
    lemmatized_tokens = [
        lemmatizer.lemmatize(token)
        for token in clean_tokens
    ]

    # 4. Stop-word Removal
    filtered_tokens = [
        token
        for token in clean_tokens
        if token not in stop_words
    ]

    print("\n1. Tokenization:")
    print(clean_tokens)

    print("\n2. Stemming:")
    print(stemmed_tokens)

    print("\n3. Lemmatization:")
    print(lemmatized_tokens)

    print("\n4. Stop-word Removal:")
    print(filtered_tokens)