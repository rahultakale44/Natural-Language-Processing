import spacy


class DependencyParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def parse(self, sentence):
        doc = self.nlp(sentence)
        return doc

    def get_dependencies(self, doc):
        dependencies = []

        for token in doc:
            dependencies.append({
                "word": token.text,
                "pos": token.pos_,
                "tag": token.tag_,
                "head": token.head.text,
                "dependency": token.dep_
            })

        return dependencies

    def display_dependencies(self, doc):
        print("\n" + "=" * 80)
        print("DEPENDENCY PARSING")
        print("=" * 80)

        print(
            f"{'Word':<20}"
            f"{'POS':<10}"
            f"{'Head':<20}"
            f"{'Dependency':<20}"
        )

        print("-" * 80)

        for token in doc:
            print(
                f"{token.text:<20}"
                f"{token.pos_:<10}"
                f"{token.head.text:<20}"
                f"{token.dep_:<20}"
            )

    def display_tree(self, doc):
        print("\n" + "=" * 80)
        print("DEPENDENCY TREE")
        print("=" * 80)

        for token in doc:
            print(
                f"{token.text} "
                f"--> {token.head.text} "
                f"[{token.dep_}]"
            )