

from typing import List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from rag.types import Chunk, Entity, Relation
from rag.graph.entity_extractor import EntityExtractor

class KGState(dict):

    pass

class KnowledgeGraphBuilder:

    def __init__(
        self,
        entity_extractor: EntityExtractor,
        neo4j_store: Optional[object] = None,
        enable_enrichment: bool = True,
    ):
        self.entity_extractor = entity_extractor
        self.neo4j_store = neo4j_store
        self.enable_enrichment = enable_enrichment
        self.graph_flow = self._build_flow()

    def _build_flow(self):

        g = StateGraph(KGState)

        g.add_node("extract", self._node_extract)
        g.add_node("normalize", self._node_normalize)

        if self.enable_enrichment:
            g.add_node("enrich", self._node_enrich)

        if self.neo4j_store:
            g.add_node("store", self._node_store)

        g.set_entry_point("extract")
        g.add_edge("extract", "normalize")

        if self.enable_enrichment:
            g.add_edge("normalize", "enrich")
            if self.neo4j_store:
                g.add_edge("enrich", "store")
                g.add_edge("store", END)
            else:
                g.add_edge("enrich", END)
        else:
            if self.neo4j_store:
                g.add_edge("normalize", "store")
                g.add_edge("store", END)
            else:
                g.add_edge("normalize", END)

        return g.compile()

    def _node_extract(self, state: KGState) -> KGState:

        chunks: List[Chunk] = state.get("chunks", [])

        if not chunks:
            state["entities"] = []
            state["relations"] = []
            return state

        print(f"[KGBuilder] Extracting entities from {len(chunks)} chunks...")

        entities, relations = self.entity_extractor.extract_from_chunks(chunks)

        state["entities"] = entities
        state["relations"] = relations
        state["chunks"] = chunks  

        print(f"[KGBuilder] Extracted {len(entities)} entities and {len(relations)} relations")

        return state

    def _node_normalize(self, state: KGState) -> KGState:

        entities: List[Entity] = state.get("entities", [])
        relations: List[Relation] = state.get("relations", [])

        print(f"[KGBuilder] Normalizing {len(entities)} entities...")

        entity_lookup = {
            (e.canonical_form.lower(), e.entity_type): e
            for e in entities
        }

        state["entity_lookup"] = entity_lookup
        state["normalized_entities"] = entities
        state["normalized_relations"] = relations

        print(f"[KGBuilder] Normalized to {len(entities)} unique entities")

        return state

    def _node_enrich(self, state: KGState) -> KGState:

        entities: List[Entity] = state.get("normalized_entities", [])

        print(f"[KGBuilder] Enriching {len(entities)} entities...")

        for entity in entities:
            occurrences = entity.metadata.get("occurrences", 1)
            entity.metadata["importance_score"] = min(1.0, occurrences / 10.0)

        state["enriched_entities"] = entities

        print(f"[KGBuilder] Enrichment complete")

        return state

    def _node_store(self, state: KGState) -> KGState:

        entities: List[Entity] = state.get("enriched_entities") or state.get("normalized_entities", [])
        relations: List[Relation] = state.get("normalized_relations", [])
        chunks: List[Chunk] = state.get("chunks", [])

        if not self.neo4j_store:
            return state

        print(f"[KGBuilder] Storing {len(entities)} entities and {len(relations)} relations in Neo4j...")

        try:

            self.neo4j_store.ensure_entity_indexes()

            self.neo4j_store.upsert_entities(entities)

            self.neo4j_store.upsert_relations(relations)

            self.neo4j_store.link_entities_to_chunks(chunks)

            print(f"[KGBuilder] Successfully stored knowledge graph")

        except Exception as e:
            print(f"[KGBuilder] Failed to store in Neo4j: {e}")

        return state

    def build(self, chunks: List[Chunk]) -> Dict[str, Any]:

        initial_state: KGState = {
            "chunks": chunks,
            "entities": [],
            "relations": [],
        }

        final_state = self.graph_flow.invoke(initial_state)

        return {
            "entities": final_state.get("enriched_entities") or final_state.get("normalized_entities", []),
            "relations": final_state.get("normalized_relations", []),
            "chunks": final_state.get("chunks", []),
            "entity_lookup": final_state.get("entity_lookup", {}),
        }

    def update_incremental(self, new_chunks: List[Chunk], existing_entities: List[Entity]) -> Dict[str, Any]:

        new_entities, new_relations = self.entity_extractor.extract_from_chunks(new_chunks)

        all_entities = existing_entities + new_entities
        resolved_entities = self.entity_extractor.resolve_entities(all_entities)

        state: KGState = {
            "chunks": new_chunks,
            "entities": resolved_entities,
            "relations": new_relations,
        }

        state = self._node_normalize(state)

        if self.enable_enrichment:
            state = self._node_enrich(state)

        if self.neo4j_store:
            state = self._node_store(state)

        return {
            "entities": state.get("enriched_entities") or state.get("normalized_entities", []),
            "relations": state.get("normalized_relations", []),
            "chunks": state.get("chunks", []),
            "entity_lookup": state.get("entity_lookup", {}),
        }

def build_kg_flow(
    entity_extractor: EntityExtractor,
    neo4j_store: Optional[object] = None,
    enable_enrichment: bool = True,
):

    builder = KnowledgeGraphBuilder(
        entity_extractor=entity_extractor,
        neo4j_store=neo4j_store,
        enable_enrichment=enable_enrichment,
    )
    return builder.graph_flow
