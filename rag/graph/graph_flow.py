from typing import Dict, List, Tuple
from langgraph.graph import StateGraph, END

from rag.graph.kg import KnowledgeGraph

class KGState(dict):
    pass

def extract_entities_relations(texts: List[str]) -> List[Tuple[str, str, str]]:

    triples: List[Tuple[str, str, str]] = []
    for t in texts:

        parts = t.split(" is ")
        if len(parts) == 2:
            subj = parts[0].split()[-1]
            obj = parts[1].split()[0]
            triples.append((subj, "is", obj))
    return triples

def build_kg_flow(kg: KnowledgeGraph):
    g = StateGraph(KGState)

    def node_extract(state: KGState) -> KGState:
        texts: List[str] = state.get("texts", [])
        triples = extract_entities_relations(texts)
        state["triples"] = triples
        return state

    def node_update(state: KGState) -> KGState:
        triples = state.get("triples", [])
        kg.add_triples(triples)
        return state

    g.add_node("extract", node_extract)
    g.add_node("update", node_update)
    g.set_entry_point("extract")
    g.add_edge("extract", "update")
    g.add_edge("update", END)

    return g.compile()
