import nltk
from nltk import CFG
from nltk.parse import ChartParser


class ConstituencyParser:
    def __init__(self):
        self.grammar = CFG.fromstring("""
            S -> NP VP
            S -> NP VP CONJ S

            NP -> DET ADJ NOUN
            NP -> DET NOUN
            NP -> DET ADJ NOUN PP
            NP -> NOUN
            NP -> PROPN
            NP -> NP PP

            VP -> VERB
            VP -> VERB NP
            VP -> VERB NP PP
            VP -> VERB ADV
            VP -> VERB S
            VP -> VERB NP S

            PP -> PREP NP

            DET -> 'the' | 'a' | 'an'
            ADJ -> 'intelligent' | 'difficult' | 'detailed' | 'complex'
            NOUN -> 'student' | 'assignment' | 'professor' | 'instructions' | 'system'
            PROPN -> 'John' | 'Mary'
            VERB -> 'completed' | 'provided' | 'designed' | 'analyzed' | 'studied'
            ADV -> 'quickly' | 'carefully'
            PREP -> 'with' | 'for' | 'in'
            CONJ -> 'because' | 'and'
        """)

        self.parser = ChartParser(self.grammar)

    def tokenize(self, sentence):
        return nltk.word_tokenize(sentence.lower())

    def parse(self, sentence):
        tokens = self.tokenize(sentence)

        try:
            trees = list(self.parser.parse(tokens))
            return trees
        except ValueError:
            return []

    def display_trees(self, trees):
        print("\n" + "=" * 80)
        print("CONSTITUENCY PARSING")
        print("=" * 80)

        if not trees:
            print("No valid constituency parse found.")
            return

        for i, tree in enumerate(trees, start=1):
            print(f"\nParse Tree {i}:")
            tree.pretty_print()

    def get_tree_string(self, trees):
        if not trees:
            return "No valid constituency parse found."

        return str(trees[0])