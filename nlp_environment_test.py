import sys


def print_heading(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def test_nltk():
    print_heading("1. NLTK TEST")

    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize

    # Download required NLTK datasets.
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)

    text = "Natural Language Processing helps computers understand human language."

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words("english"))

    filtered_words = [
        word
        for word in tokens
        if word.lower() not in stop_words and word.isalpha()
    ]

    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    print("NLTK version:", nltk.__version__)
    print("Original text:", text)
    print("Tokens:", tokens)
    print("After stop-word removal:", filtered_words)
    print("Stemmed words:", stemmed_words)

    print("\nNLTK test passed successfully.")


def test_spacy():
    print_heading("2. SPACY TEST")

    import spacy

    nlp = spacy.load("en_core_web_sm")

    text = "Microsoft opened a new office in Pune, India in 2025."

    document = nlp(text)

    print("spaCy version:", spacy.__version__)
    print("Original text:", text)

    print("\nToken analysis:")

    for token in document:
        print(
            f"Token: {token.text:<12} "
            f"Lemma: {token.lemma_:<12} "
            f"POS: {token.pos_}"
        )

    print("\nNamed entities:")

    for entity in document.ents:
        print(f"{entity.text} -> {entity.label_}")

    print("\nspaCy test passed successfully.")


def test_gensim():
    print_heading("3. GENSIM TEST")

    import gensim
    from gensim.models import Word2Vec

    sentences = [
        ["natural", "language", "processing", "uses", "text"],
        ["machine", "learning", "uses", "data"],
        ["deep", "learning", "uses", "neural", "networks"],
        ["language", "models", "learn", "from", "text"],
        ["nlp", "is", "part", "of", "artificial", "intelligence"],
        ["transformers", "perform", "many", "language", "tasks"],
    ]

    model = Word2Vec(
        sentences=sentences,
        vector_size=20,
        window=3,
        min_count=1,
        workers=1,
        epochs=100,
        seed=42,
    )

    print("Gensim version:", gensim.__version__)
    print("Vocabulary:", list(model.wv.index_to_key))

    print("\nVector for the word 'language':")
    print(model.wv["language"])

    print("\nWords similar to 'language':")

    similar_words = model.wv.most_similar("language", topn=3)

    for word, score in similar_words:
        print(f"{word}: {score:.4f}")

    print("\nGensim test passed successfully.")


def test_transformers():
    print_heading("4. HUGGING FACE TRANSFORMERS TEST")

    import torch
    import transformers
    from transformers import pipeline

    print("Transformers version:", transformers.__version__)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    sentiment_analyzer = pipeline(
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,
    )

    sentences = [
        "I enjoyed learning natural language processing.",
        "The application was difficult to use and produced poor results.",
    ]

    results = sentiment_analyzer(sentences)

    for sentence, result in zip(sentences, results):
        print("\nSentence:", sentence)
        print("Sentiment:", result["label"])
        print("Confidence:", round(result["score"], 4))

    print("\nTransformers test passed successfully.")


def main():
    print_heading("PYTHON NLP ENVIRONMENT VERIFICATION")

    print("Python version:", sys.version)

    tests = [
        ("NLTK", test_nltk),
        ("spaCy", test_spacy),
        ("Gensim", test_gensim),
        ("Hugging Face Transformers", test_transformers),
    ]

    passed_tests = 0

    for library_name, test_function in tests:
        try:
            test_function()
            passed_tests += 1

        except Exception as error:
            print(f"\n{library_name} test failed.")
            print("Error:", error)

    print_heading("FINAL RESULT")

    print(f"Tests passed: {passed_tests}/{len(tests)}")

    if passed_tests == len(tests):
        print("All NLP libraries are installed and working successfully.")
    else:
        print("Some tests failed. Check the error messages above.")


if __name__ == "__main__":
    main()