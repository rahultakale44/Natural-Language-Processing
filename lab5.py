# ============================================================
# ASSIGNMENT 5
# MORPHOLOGICAL ANALYSIS AND POS TAGGING SYSTEM
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import spacy
import nltk
import pandas as pd
import matplotlib.pyplot as plt

from nltk.stem import PorterStemmer
from collections import Counter


# ============================================================
# 2. LOAD SPACY ENGLISH MODEL
# ============================================================

nlp = spacy.load("en_core_web_sm")


# ============================================================
# 3. INPUT TEXT
# ============================================================

text = """
Natural Language Processing is transforming modern technology.
Machine learning and artificial intelligence are changing the world.
Researchers are developing intelligent systems for automation.
"""


# Process text using spaCy
doc = nlp(text)


# Initialize Porter Stemmer
stemmer = PorterStemmer()


# ============================================================
# 4. WORD, STEM AND LEMMA
# ============================================================

words = []
stems = []
lemmas = []


for token in doc:

    # Ignore spaces and punctuation
    if not token.is_space and not token.is_punct:

        words.append(token.text)

        stems.append(
            stemmer.stem(token.text)
        )

        lemmas.append(
            token.lemma_
        )


# Create DataFrame
word_stem_lemma = pd.DataFrame({

    "Word": words,
    "Stem": stems,
    "Lemma": lemmas

})


# Display result
print("\n" + "=" * 80)
print("WORD, STEM AND LEMMA")
print("=" * 80)

print(
    word_stem_lemma.to_string(index=True)
)


# ============================================================
# 5. PART OF SPEECH TAGGING
# ============================================================

pos_data = []


for token in doc:

    # Ignore spaces
    if not token.is_space:

        pos_data.append({

            "Word": token.text,

            "POS Tag": token.tag_

        })


# Create DataFrame
pos_df = pd.DataFrame(pos_data)


# Display result
print("\n" + "=" * 80)
print("PART OF SPEECH TAGGING")
print("=" * 80)

print(
    pos_df.to_string(index=True)
)


# ============================================================
# 6. WORD FREQUENCY
# ============================================================

# Get all tokens except spaces
word_tokens = [

    token.text

    for token in doc

    if not token.is_space
]


# Count word frequency
word_frequency = Counter(word_tokens)


# Convert to DataFrame
frequency_df = pd.DataFrame(

    word_frequency.items(),

    columns=[
        "Word",
        "Frequency"
    ]

)


# Sort by frequency
frequency_df = frequency_df.sort_values(

    by="Frequency",

    ascending=False

)


# Display result
print("\n" + "=" * 80)
print("WORD FREQUENCY")
print("=" * 80)

print(
    frequency_df.to_string(index=False)
)


# ============================================================
# 7. POS TAG FREQUENCY
# ============================================================

# Count POS tags
pos_frequency = Counter(

    token.tag_

    for token in doc

    if not token.is_space
    and not token.is_punct

)


# Convert to DataFrame
pos_frequency_df = pd.DataFrame(

    pos_frequency.items(),

    columns=[
        "POS Tag",
        "Frequency"
    ]

)


# Sort by frequency
pos_frequency_df = pos_frequency_df.sort_values(

    by="Frequency",

    ascending=False

)


# Display result
print("\n" + "=" * 80)
print("POS TAG FREQUENCY")
print("=" * 80)

print(
    pos_frequency_df.to_string(index=False)
)


# ============================================================
# 8. GROUP LEXICAL WORDS BY POS TAG
# ============================================================

# Dictionary:
#
# POS Tag  ->  Words belonging to that POS
#
# Example:
#
# NN  -> Processing, technology, learning, intelligence,
#        world, systems, automation
#
# JJ  -> modern, artificial, intelligent

pos_words = {}


for token in doc:

    # Ignore spaces and punctuation
    if not token.is_space and not token.is_punct:

        tag = token.tag_


        # Create list if POS tag is not already present
        if tag not in pos_words:

            pos_words[tag] = []


        # Add word to corresponding POS tag
        pos_words[tag].append(token.text)


# Remove duplicate words while maintaining order
for tag in pos_words:

    pos_words[tag] = list(
        dict.fromkeys(
            pos_words[tag]
        )
    )


# Display grouped lexical words
print("\n" + "=" * 80)
print("LEXICAL WORDS GROUPED BY POS TAG")
print("=" * 80)


for tag in pos_frequency_df["POS Tag"]:

    if tag in pos_words:

        words_for_tag = ", ".join(
            pos_words[tag]
        )

        print(
            f"{tag:<6} : {words_for_tag}"
        )


# ============================================================
# 9. POS TAG FREQUENCY GRAPH
#    WITH LEXICAL WORDS DISPLAYED VERTICALLY
# ============================================================

graph_data = []


for tag in pos_frequency_df["POS Tag"]:

    if tag in pos_words:

        # Frequency of the POS tag
        frequency = len([

            token

            for token in doc

            if not token.is_space
            and not token.is_punct
            and token.tag_ == tag

        ])

        # Get lexical words
        words_for_tag = pos_words[tag]

        graph_data.append({

            "POS": tag,

            "Frequency": frequency,

            "Words": words_for_tag

        })


graph_df = pd.DataFrame(graph_data)


# ============================================================
# CREATE LARGE GRAPH
# ============================================================

plt.figure(
    figsize=(20, 12)
)


# Create ONE bar for each POS tag
bars = plt.bar(

    graph_df["POS"],

    graph_df["Frequency"]

)


# ============================================================
# TITLE AND AXIS LABELS
# ============================================================

plt.title(

    "POS Tag Frequency Distribution with Lexical Words",

    fontsize=18,

    pad=20

)


plt.xlabel(

    "POS Tags and Lexical Words",

    fontsize=13

)


plt.ylabel(

    "Frequency",

    fontsize=13

)


# ============================================================
# DISPLAY POS TAG ABOVE EACH BAR
# ============================================================

for bar in bars:

    height = bar.get_height()

    plt.text(

        bar.get_x()
        + bar.get_width() / 2,

        height + 0.1,

        str(int(height)),

        ha="center",

        va="bottom",

        fontsize=11

    )


# ============================================================
# DISPLAY LEXICAL WORDS VERTICALLY
# UNDER EACH POS TAG
# ============================================================

labels = []


for _, row in graph_df.iterrows():

    pos = row["POS"]

    words_for_tag = row["Words"]


    # Put every word on a separate line
    word_text = "\n".join(words_for_tag)


    # POS tag + vertical word list
    label = (

        pos
        + "\n"
        + word_text

    )


    labels.append(label)


# Apply labels
plt.xticks(

    range(len(labels)),

    labels,

    rotation=0,

    fontsize=10

)


# ============================================================
# ADD EXTRA SPACE AT THE BOTTOM
# ============================================================

plt.subplots_adjust(

    bottom=0.35,

    top=0.90,

    left=0.06,

    right=0.98

)


# Display graph
plt.show()


# ============================================================
# 10. MORPHOLOGICAL ANALYSIS
# ============================================================

morph_data = []


for token in doc:

    # Ignore spaces
    if not token.is_space:

        # ------------------------------------
        # TENSE
        # ------------------------------------

        tense = ", ".join(

            token.morph.get("Tense")

        )


        # ------------------------------------
        # NUMBER
        # ------------------------------------

        number = ", ".join(

            token.morph.get("Number")

        )


        # ------------------------------------
        # PERSON
        # ------------------------------------

        person = ", ".join(

            token.morph.get("Person")

        )


        # ------------------------------------
        # Replace empty values with "-"
        # ------------------------------------

        if tense == "":
            tense = "-"


        if number == "":
            number = "-"


        if person == "":
            person = "-"


        # ------------------------------------
        # Store morphological information
        # ------------------------------------

        morph_data.append({

            "Word": token.text,

            "Lemma": token.lemma_,

            "POS": token.pos_,

            "Tag": token.tag_,

            "Tense": tense,

            "Number": number,

            "Person": person

        })


# Create DataFrame
morph_df = pd.DataFrame(
    morph_data
)


# Display result
print("\n" + "=" * 100)
print("MORPHOLOGICAL ANALYSIS")
print("=" * 100)

print(
    morph_df.to_string(index=True)
)