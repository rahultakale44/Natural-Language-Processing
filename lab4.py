import re
from collections import Counter
import matplotlib.pyplot as plt

# Input sentences
text = """
Natural language processing is a field of artificial intelligence.
Machine learning improves natural language processing systems.
Artificial intelligence powers modern NLP applications.
"""

# Convert text to lowercase and extract words
words = re.findall(r'\b[a-z]+\b', text.lower())

# -------------------------
# 1. Bag of Words (BoW)
# -------------------------
bow = Counter(words)

print("========== BAG OF WORDS ==========")
for word, count in bow.items():
    print(f"{word}: {count}")

# -------------------------
# 2. Bigrams
# -------------------------
bigrams = list(zip(words, words[1:]))
bigram_count = Counter(bigrams)

print("\n========== BIGRAMS ==========")
for bigram, count in bigram_count.items():
    print(f"{bigram}: {count}")

# -------------------------
# 3. Trigrams
# -------------------------
trigrams = list(zip(words, words[1:], words[2:]))
trigram_count = Counter(trigrams)

print("\n========== TRIGRAMS ==========")
for trigram, count in trigram_count.items():
    print(f"{trigram}: {count}")

# -------------------------
# 4. Frequency Distribution Plot
# -------------------------
top_words = bow.most_common()

word_names = [item[0] for item in top_words]
word_counts = [item[1] for item in top_words]

plt.figure(figsize=(12, 6))
plt.bar(word_names, word_counts)

plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Word Frequency Distribution")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()