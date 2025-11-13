import networkx as nx
from typing import List, Tuple

class KnowledgeGraph:
    def __init__(self):
        self.g = nx.MultiDiGraph()

    def add_triple(self, subj: str, pred: str, obj: str):
        self.g.add_edge(subj, obj, label=pred)

    def add_triples(self, triples: List[Tuple[str, str, str]]):
        for s, p, o in triples:
            self.add_triple(s, p, o)

    def neighbors(self, node: str) -> List[str]:
        return list(self.g.neighbors(node))
