

import numpy as np
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense




text = """
I love natural language processing
I love machine learning
I love deep learning
machine learning is interesting
deep learning is powerful
natural language processing is interesting
"""


words = text.lower().split()

n = 3
ngrams = []

for i in range(len(words) - n + 1):
    ngrams.append(tuple(words[i:i+n]))

counts = Counter(ngrams)

print("N-GRAM MODEL")
print("-------------------------")

print("Most frequent trigrams:")

for gram, count in counts.most_common(5):
    print(gram, "->", count)


def predict_ngram(word1, word2):

    candidates = []

    for gram, count in counts.items():

        if gram[0] == word1 and gram[1] == word2:
            candidates.append((gram[2], count))

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    return "No prediction"


prediction = predict_ngram("machine", "learning")

print("\nN-Gram Prediction:")
print("machine learning ->", prediction)




tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

input_sequences = []

for sentence in text.strip().split("\n"):

    sequence = tokenizer.texts_to_sequences([sentence])[0]

    for i in range(1, len(sequence)):
        input_sequences.append(sequence[:i+1])


max_sequence_len = max(
    len(sequence) for sequence in input_sequences
)

input_sequences = pad_sequences(
    input_sequences,
    maxlen=max_sequence_len,
    padding="pre"
)

X = input_sequences[:, :-1]
y = input_sequences[:, -1]




model = Sequential()

model.add(
    Embedding(
        total_words,
        50,
        input_length=max_sequence_len - 1
    )
)

model.add(LSTM(100))

model.add(Dense(total_words, activation="softmax"))

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)


print("\nTraining Neural Language Model...")

model.fit(
    X,
    y,
    epochs=100,
    verbose=0
)



def predict_next_word(seed_text):

    sequence = tokenizer.texts_to_sequences(
        [seed_text]
    )[0]

    sequence = pad_sequences(
        [sequence],
        maxlen=max_sequence_len - 1,
        padding="pre"
    )

    prediction = model.predict(
        sequence,
        verbose=0
    )

    predicted_index = np.argmax(prediction)

    for word, index in tokenizer.word_index.items():

        if index == predicted_index:
            return word

    return "Unknown"


seed = "machine learning"

predicted_word = predict_next_word(seed)

print("\nNEURAL LANGUAGE MODEL")
print("-------------------------")

print("Input:", seed)

print("Predicted Next Word:", predicted_word)