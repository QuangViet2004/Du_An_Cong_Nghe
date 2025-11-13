

import numpy as np
from typing import List, Tuple, Optional, Set, Dict
from collections import defaultdict

from rag.types import Chunk, Entity
from rag.graph.kg_query import KnowledgeGraphQuery

class KGEnhancedRetriever:

    def __init__(
        self,
        base_indexer,  
        neo4j_store,  
        text_embedder=None,  
        retrieval_mode: str = "hybrid",  
        expansion_hops: int = 2,
        entity_boost_weight: float = 0.3,
        centrality_weight: float = 0.2,
    ):

        self.base_indexer = base_indexer
        self.neo4j = neo4j_store
        self.text_embedder = text_embedder
        self.retrieval_mode = retrieval_mode
        self.expansion_hops = expansion_hops
        self.entity_boost_weight = entity_boost_weight
        self.centrality_weight = centrality_weight

        self.kg_query = KnowledgeGraphQuery(neo4j_store) if neo4j_store else None

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.6,
        **kwargs
    ) -> List[Tuple[Chunk, float, float, float]]:

        if self.retrieval_mode == "baseline" or not self.kg_query:
            return self.base_indexer.search(query, top_k, alpha)

        elif self.retrieval_mode == "entity_aware":
            return self._entity_aware_search(query, top_k, alpha)

        elif self.retrieval_mode == "graph_guided":
            return self._graph_guided_search(query, top_k, alpha)

        elif self.retrieval_mode == "hybrid":
            return self._hybrid_kg_search(query, top_k, alpha)

        else:

            return self.base_indexer.search(query, top_k, alpha)

    def _entity_aware_search(
        self,
        query: str,
        top_k: int,
        alpha: float
    ) -> List[Tuple[Chunk, float, float, float]]:

        initial_k = min(top_k * 3, len(self.base_indexer.chunks))
        base_results = self.base_indexer.search(query, initial_k, alpha)

        if not self.text_embedder or not base_results:
            return base_results[:top_k]

        query_entities = self.kg_query.extract_entities_from_query(
            query,
            self.text_embedder
        )

        if not query_entities:

            return base_results[:top_k]

        print(f"[KGRetriever] Found {len(query_entities)} query entities: {[e.canonical_form for e in query_entities]}")

        query_entity_keys = {
            (e.canonical_form.lower(), e.entity_type)
            for e in query_entities
        }

        boosted_results = []
        for chunk, combined_score, sem_score, bm25_score in base_results:
            boost = 0.0

            if chunk.entities:

                chunk_entity_keys = {
                    (e.canonical_form.lower(), e.entity_type)
                    for e in chunk.entities
                }

                matches = query_entity_keys.intersection(chunk_entity_keys)
                if matches:

                    boost = len(matches) / max(len(query_entity_keys), 1.0)

            boosted_combined = combined_score + (boost * self.entity_boost_weight)
            boosted_results.append((chunk, boosted_combined, sem_score, bm25_score))

        boosted_results.sort(key=lambda x: x[1], reverse=True)
        return boosted_results[:top_k]

    def _graph_guided_search(
        self,
        query: str,
        top_k: int,
        alpha: float
    ) -> List[Tuple[Chunk, float, float, float]]:

        initial_k = min(top_k, len(self.base_indexer.chunks))
        initial_results = self.base_indexer.search(query, initial_k, alpha)

        if not initial_results:
            return []

        seed_entities: List[Entity] = []
        for chunk, _, _, _ in initial_results[:3]:  
            if chunk.entities:
                seed_entities.extend(chunk.entities)

        if not seed_entities:

            return initial_results

        print(f"[KGRetriever] Expanding from {len(seed_entities)} seed entities")

        related_entities = self.kg_query.find_related_entities(
            seed_entities,
            max_hops=self.expansion_hops,
            min_importance=0.3
        )

        if not related_entities:
            return initial_results

        print(f"[KGRetriever] Found {len(related_entities)} related entities")

        expansion_chunks: Dict[int, Tuple[Chunk, float]] = {}  

        for entity_props in related_entities[:10]:  
            chunk_props_list = self.neo4j.get_entity_chunks(
                entity_props["canonical_form"],
                entity_props["type"],
                limit=2
            )

            for chunk_props in chunk_props_list:

                source = chunk_props.get("source")
                page = chunk_props.get("page")
                chunk_idx = chunk_props.get("idx")

                if chunk_idx is not None and 0 <= chunk_idx < len(self.base_indexer.chunks):
                    chunk = self.base_indexer.chunks[chunk_idx]
                    if chunk.source == source and chunk.page == page:

                        relevance = entity_props.get("importance_score", 0.5)
                        if chunk_idx not in expansion_chunks:
                            expansion_chunks[chunk_idx] = (chunk, relevance)

        all_results: Dict[int, Tuple[Chunk, float, float, float]] = {}

        for chunk, combined, sem, bm25 in initial_results:

            for idx, c in enumerate(self.base_indexer.chunks):
                if c is chunk:
                    all_results[idx] = (chunk, combined, sem, bm25)
                    break

        for chunk_idx, (chunk, relevance) in expansion_chunks.items():
            if chunk_idx not in all_results:

                pseudo_combined = relevance * 0.5  
                pseudo_sem = relevance * 0.5
                pseudo_bm25 = 0.0
                all_results[chunk_idx] = (chunk, pseudo_combined, pseudo_sem, pseudo_bm25)

        sorted_results = sorted(
            all_results.values(),
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_results[:top_k]

    def _hybrid_kg_search(
        self,
        query: str,
        top_k: int,
        alpha: float
    ) -> List[Tuple[Chunk, float, float, float]]:

        query_entities = []
        if self.text_embedder:
            query_entities = self.kg_query.extract_entities_from_query(
                query,
                self.text_embedder
            )

        initial_k = min(top_k * 4, len(self.base_indexer.chunks))
        base_results = self.base_indexer.search(query, initial_k, alpha)

        if not base_results:
            return []

        baseline_chunks = {id(chunk): (chunk, combined, sem, bm25)
                          for chunk, combined, sem, bm25 in base_results}

        entity_scores: Dict[int, float] = defaultdict(float)

        if query_entities:
            query_entity_keys = {
                (e.canonical_form.lower(), e.entity_type)
                for e in query_entities
            }

            for chunk, _, _, _ in base_results:
                if chunk.entities:
                    chunk_entity_keys = {
                        (e.canonical_form.lower(), e.entity_type)
                        for e in chunk.entities
                    }
                    matches = query_entity_keys.intersection(chunk_entity_keys)
                    if matches:
                        chunk_id = id(chunk)
                        entity_scores[chunk_id] = len(matches) / max(len(query_entity_keys), 1.0)

        seed_entities: List[Entity] = []
        for chunk, _, _, _ in base_results[:3]:
            if chunk.entities:
                seed_entities.extend(chunk.entities)

        expansion_chunks: Dict[int, Tuple[Chunk, float]] = {}

        if seed_entities:
            related_entities = self.kg_query.find_related_entities(
                seed_entities,
                max_hops=self.expansion_hops,
                min_importance=0.3
            )

            for entity_props in related_entities[:8]:
                chunk_props_list = self.neo4j.get_entity_chunks(
                    entity_props["canonical_form"],
                    entity_props["type"],
                    limit=2
                )

                for chunk_props in chunk_props_list:
                    chunk_idx = chunk_props.get("idx")
                    if chunk_idx is not None and 0 <= chunk_idx < len(self.base_indexer.chunks):
                        chunk = self.base_indexer.chunks[chunk_idx]
                        relevance = entity_props.get("importance_score", 0.5)
                        if chunk_idx not in expansion_chunks:
                            expansion_chunks[chunk_idx] = (chunk, relevance)

        centrality_scores: Dict[int, float] = defaultdict(float)

        for chunk, _, _, _ in base_results:
            if chunk.entities:

                chunk_id = id(chunk)
                centrality_scores[chunk_id] = min(1.0, len(chunk.entities) / 5.0)

        final_results = []

        for chunk, combined, sem, bm25 in base_results:
            chunk_id = id(chunk)

            entity_boost = entity_scores.get(chunk_id, 0.0) * self.entity_boost_weight
            centrality_boost = centrality_scores.get(chunk_id, 0.0) * self.centrality_weight

            final_combined = combined + entity_boost + centrality_boost

            final_results.append((chunk, final_combined, sem, bm25))

        for chunk_idx, (chunk, relevance) in expansion_chunks.items():
            chunk_id = id(chunk)
            if chunk_id not in baseline_chunks:
                pseudo_combined = relevance * 0.4
                pseudo_sem = relevance * 0.4
                pseudo_bm25 = 0.0

                entity_boost = entity_scores.get(chunk_id, 0.0) * self.entity_boost_weight
                centrality_boost = centrality_scores.get(chunk_id, 0.0) * self.centrality_weight

                final_combined = pseudo_combined + entity_boost + centrality_boost

                final_results.append((chunk, final_combined, pseudo_sem, pseudo_bm25))

        final_results.sort(key=lambda x: x[1], reverse=True)

        print(f"[KGRetriever] Hybrid search: {len(base_results)} baseline + {len(expansion_chunks)} expanded = {len(final_results)} total")

        return final_results[:top_k]

    def get_entity_context_for_results(
        self,
        results: List[Tuple[Chunk, float, float, float]]
    ) -> Dict[str, Any]:

        if not self.kg_query:
            return {}

        all_entities: List[Entity] = []
        for chunk, _, _, _ in results:
            if chunk.entities:
                all_entities.extend(chunk.entities)

        if not all_entities:
            return {}

        unique_entities = list({
            (e.canonical_form, e.entity_type): e
            for e in all_entities
        }.values())

        importance_scores = self.kg_query.get_entity_importance_scores(unique_entities)

        subgraph = self.kg_query.get_entity_subgraph(
            unique_entities[:10],  
            include_neighbors=True
        )

        return {
            "entities": unique_entities,
            "importance_scores": importance_scores,
            "subgraph": subgraph,
        }
