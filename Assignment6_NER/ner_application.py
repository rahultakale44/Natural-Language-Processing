# ============================================================
# Assignment 6
# Named Entity Recognition (NER) Application using spaCy
# ============================================================

import spacy
import pandas as pd


# ------------------------------------------------------------
# Load spaCy English Language Model
# ------------------------------------------------------------

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("\nSpaCy English model is not installed.")
    print("Run this command first:")
    print("python -m spacy download en_core_web_sm")
    exit()


# ------------------------------------------------------------
# Input Text
# ------------------------------------------------------------

text = """
Sundar Pichai is the CEO of Google and was born in Chennai, India.
Microsoft Corporation opened a new research center in Hyderabad on 12 March 2025.
Elon Musk announced a new investment of $2 billion in an artificial intelligence project.
Apple Inc. is looking at buying U.K. startup for $1 billion.
Tim Cook announced the potential acquisition on Monday in California.
"""


# ------------------------------------------------------------
# Process Text Using spaCy
# ------------------------------------------------------------

doc = nlp(text)


# ------------------------------------------------------------
# Entity Label Meanings
# ------------------------------------------------------------

entity_meanings = {
    "PERSON": "Person",
    "ORG": "Organization",
    "GPE": "Geopolitical Entity / Location",
    "LOC": "Location",
    "DATE": "Date",
    "TIME": "Time",
    "MONEY": "Monetary Value",
    "PRODUCT": "Product",
    "EVENT": "Event",
    "NORP": "Nationality / Religious / Political Group",
    "FAC": "Facility",
    "LAW": "Law",
    "LANGUAGE": "Language"
}


# ------------------------------------------------------------
# Extract Named Entities
# ------------------------------------------------------------

entities = []

for ent in doc.ents:

    label = ent.label_

    meaning = entity_meanings.get(
        label,
        "Other Entity"
    )

    entities.append({
        "Entity": ent.text,
        "Label": label,
        "Meaning": meaning
    })


# ------------------------------------------------------------
# Create DataFrame
# ------------------------------------------------------------

df = pd.DataFrame(entities)


# ------------------------------------------------------------
# Display Application Header
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("           NAMED ENTITY RECOGNITION (NER) APPLICATION")
print("                         USING SPACY")
print("=" * 75)


# ------------------------------------------------------------
# Display Input Text
# ------------------------------------------------------------

print("\nINPUT TEXT:")
print("-" * 75)

print(text)


# ------------------------------------------------------------
# Display Detected Entities
# ------------------------------------------------------------

print("\nDETECTED NAMED ENTITIES:")
print("-" * 75)


if not df.empty:
    print(df.to_string(index=False))
else:
    print("No named entities were detected.")


# ------------------------------------------------------------
# Group Entities by Category
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("ENTITIES GROUPED BY CATEGORY")
print("=" * 75)


grouped_entities = {}

for ent in doc.ents:

    label = ent.label_

    if label not in grouped_entities:
        grouped_entities[label] = []

    grouped_entities[label].append(ent.text)


for label, entity_list in grouped_entities.items():

    meaning = entity_meanings.get(
        label,
        "Other Entity"
    )

    print(f"\n{label} ({meaning})")

    for entity in entity_list:
        print(f"  • {entity}")


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("NER SUMMARY")
print("=" * 75)

print(f"\nTotal Named Entities Detected: {len(doc.ents)}")


print("\nEntity Count by Category:")

for label, entity_list in grouped_entities.items():

    meaning = entity_meanings.get(
        label,
        "Other Entity"
    )

    print(
        f"{label} ({meaning}): "
        f"{len(entity_list)}"
    )


# ------------------------------------------------------------
# Program End
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("                 PROGRAM EXECUTION COMPLETED")
print("=" * 75)