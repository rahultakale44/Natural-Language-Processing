import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

# Required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("stopwords")

document = (
    "Natural Language Processing is transforming "
    "Artificial Intelligence."
)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# 1. Tokenization
tokens = word_tokenize(document)

# Lowercase conversion and punctuation removal
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

# 4. Stop-word removal
filtered_tokens = [
    token
    for token in clean_tokens
    if token not in stop_words
]

print("Original Sentence:")
print(document)

print("\n1. Tokenization:")
print(clean_tokens)

print("\n2. Stemming:")
print(stemmed_tokens)

print("\n3. Lemmatization:")
print(lemmatized_tokens)

print("\n4. Stop-word Removal:")
print(filtered_tokens)