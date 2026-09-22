import os
import matplotlib.pyplot as plt
import networkx as nx


class ParserVisualizer:

    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def visualize_dependency_tree(self, doc, filename="dependency_tree.png"):
        graph = nx.DiGraph()

        for token in doc:
            graph.add_node(token.i, label=token.text)

            if token.head.i != token.i:
                graph.add_edge(
                    token.head.i,
                    token.i,
                    label=token.dep_
                )

        positions = self._hierarchical_layout(graph)

        plt.figure(figsize=(16, 8))

        nx.draw(
            graph,
            positions,
            with_labels=True,
            labels={
                node: graph.nodes[node]["label"]
                for node in graph.nodes
            },
            node_size=2500,
            font_size=10,
            arrows=True
        )

        edge_labels = {
            (source, target): graph.edges[source, target]["label"]
            for source, target in graph.edges
        }

        nx.draw_networkx_edge_labels(
            graph,
            positions,
            edge_labels=edge_labels,
            font_size=9
        )

        plt.title("Dependency Parse Tree")
        plt.axis("off")
        plt.tight_layout()

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

        return path

    def visualize_constituency_tree(
        self,
        tree,
        filename="constituency_tree.png"
    ):
        plt.figure(figsize=(18, 10))

        tree.draw()

        path = os.path.join(self.output_dir, filename)

        # tree.draw() opens an interactive window.
        # The tree itself can be viewed from the NLTK interface.
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

        return path

    def _hierarchical_layout(self, graph):
        levels = {}

        roots = [
            node for node in graph.nodes
            if graph.in_degree(node) == 0
        ]

        def assign_level(node, level):
            levels[node] = level

            for child in graph.successors(node):
                assign_level(child, level + 1)

        for root in roots:
            assign_level(root, 0)

        positions = {}

        level_nodes = {}

        for node, level in levels.items():
            level_nodes.setdefault(level, []).append(node)

        for level, nodes in level_nodes.items():
            count = len(nodes)

            for index, node in enumerate(nodes):
                x = (index - (count - 1) / 2) * 3
                y = -level * 2

                positions[node] = (x, y)

        return positions