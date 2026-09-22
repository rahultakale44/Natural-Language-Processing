from dependency_parser import DependencyParser
from constituency_parser import ConstituencyParser
from visualizer import ParserVisualizer


def print_header():
    print("\n" + "=" * 80)
    print("        SYNTACTIC ANALYSIS SYSTEM")
    print("        Constituency + Dependency Parsing")
    print("=" * 80)


def main():
    print_header()

    sentence = input("\nEnter a sentence: ").strip()

    if not sentence:
        print("Error: Please enter a sentence.")
        return

    # Initialize parsers
    dependency_parser = DependencyParser()
    constituency_parser = ConstituencyParser()
    visualizer = ParserVisualizer()

    # --------------------------------------------------
    # DEPENDENCY PARSING
    # --------------------------------------------------
    print("\nProcessing Dependency Parsing...")

    dependency_doc = dependency_parser.parse(sentence)

    dependency_parser.display_dependencies(dependency_doc)
    dependency_parser.display_tree(dependency_doc)

    # Save dependency visualization
    try:
        dependency_image = visualizer.visualize_dependency_tree(
            dependency_doc
        )
        print(f"\nDependency tree saved to: {dependency_image}")
    except Exception as error:
        print(f"\nDependency visualization error: {error}")

    # --------------------------------------------------
    # CONSTITUENCY PARSING
    # --------------------------------------------------
    print("\nProcessing Constituency Parsing...")

    constituency_trees = constituency_parser.parse(sentence)

    constituency_parser.display_trees(constituency_trees)

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------
    print("\n" + "=" * 80)
    print("PARSING SUMMARY")
    print("=" * 80)

    print(f"Sentence: {sentence}")
    print(f"Tokens: {len(dependency_doc)}")
    print(
        f"Constituency parses found: "
        f"{len(constituency_trees)}"
    )

    if constituency_trees:
        print("\nConstituency tree:")
        print(constituency_parser.get_tree_string(constituency_trees))

    print("\n" + "=" * 80)
    print("Parsing completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()