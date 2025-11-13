

from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict

from rag.types import Entity, Chunk

class KnowledgeGraphQuery:

    def __init__(self, neo4j_store):

        self.neo4j = neo4j_store

    def find_related_entities(
        self,
        seed_entities: List[Entity],
        max_hops: int = 2,
        min_importance: float = 0.0
    ) -> List[Dict]:

        if not seed_entities:
            return []

        all_related: Dict[Tuple[str, str], Dict] = {}

        for entity in seed_entities:
            neighbors = self.neo4j.get_entity_neighbors(
                entity.canonical_form,
                entity.entity_type,
                hops=max_hops
            )

            for neighbor in neighbors:
                importance = neighbor.get("importance_score", 0.0)
                if importance < min_importance:
                    continue

                key = (neighbor["canonical_form"], neighbor["type"])
                if key not in all_related:
                    all_related[key] = neighbor

        return list(all_related.values())

    def expand_context_with_entities(
        self,
        initial_chunks: List[Chunk],
        chunk_indices: List[int],
        expansion_hops: int = 2,
        max_additional_chunks: int = 5
    ) -> Tuple[List[int], List[Entity]]:

        seed_entities: List[Entity] = []
        for idx in chunk_indices:
            if idx < len(initial_chunks):
                chunk = initial_chunks[idx]
                if chunk.entities:
                    seed_entities.extend(chunk.entities)

        if not seed_entities:
            return chunk_indices, []

        related_entities = self.find_related_entities(
            seed_entities,
            max_hops=expansion_hops,
            min_importance=0.3
        )

        additional_chunk_keys: Set[Tuple[str, int]] = set()
        for entity_props in related_entities[:10]:  
            chunks = self.neo4j.get_entity_chunks(
                entity_props["canonical_form"],
                entity_props["type"],
                limit=3
            )

            for chunk_props in chunks:
                source = chunk_props.get("source")
                page = chunk_props.get("page")
                if source and page is not None:
                    additional_chunk_keys.add((source, page))

        related_entity_objs = []
        for props in related_entities:
            entity = Entity(
                name=props.get("name", ""),
                canonical_form=props["canonical_form"],
                entity_type=props["type"],
                confidence=props.get("confidence", 0.8),
                description=props.get("description"),
            )
            related_entity_objs.append(entity)

        return chunk_indices, related_entity_objs

    def get_entity_context(self, entity: Entity, context_size: int = 3) -> Dict:

        chunks = self.neo4j.get_entity_chunks(
            entity.canonical_form,
            entity.entity_type,
            limit=context_size
        )

        neighbors = self.neo4j.get_entity_neighbors(
            entity.canonical_form,
            entity.entity_type,
            hops=1
        )

        return {
            "entity": entity,
            "chunks": chunks,
            "neighbors": neighbors[:context_size * 2],
            "description": entity.description,
        }

    def extract_entities_from_query(
        self,
        query: str,
        text_embedder=None
    ) -> List[Entity]:

        entities_with_scores = self.neo4j.find_entities_by_name(query, limit=10)

        matched_entities = []
        for entity_props, score in entities_with_scores:
            entity = Entity(
                name=entity_props.get("name", ""),
                canonical_form=entity_props["canonical_form"],
                entity_type=entity_props["type"],
                confidence=float(score),
                description=entity_props.get("description"),
            )
            matched_entities.append(entity)

        if text_embedder and matched_entities:
            try:
                query_emb = text_embedder.encode_query(query)
                semantic_matches = self.neo4j.entity_vector_search(query_emb, top_k=5)

                for entity_props, score in semantic_matches:

                    key = (entity_props["canonical_form"], entity_props["type"])
                    existing_keys = {(e.canonical_form, e.entity_type) for e in matched_entities}

                    if key not in existing_keys:
                        entity = Entity(
                            name=entity_props.get("name", ""),
                            canonical_form=entity_props["canonical_form"],
                            entity_type=entity_props["type"],
                            confidence=float(score),
                            description=entity_props.get("description"),
                        )
                        matched_entities.append(entity)
            except Exception as e:
                print(f"[KGQuery] Semantic entity search failed: {e}")

        return matched_entities

    def get_entity_importance_scores(self, entities: List[Entity]) -> Dict[Entity, float]:

        scores = {}

        for entity in entities:

            if entity.metadata and "importance_score" in entity.metadata:
                scores[entity] = entity.metadata["importance_score"]
            else:

                scores[entity] = entity.confidence * 0.5

        return scores

    def find_connection_path(
        self,
        entity1: Entity,
        entity2: Entity,
        max_depth: int = 3
    ) -> Optional[List[Dict]]:

        query = f"""
        MATCH path = shortestPath(
            (e1:Entity {{canonical_form: $e1_canonical, type: $e1_type}})-[:RELATES_TO*1..{max_depth}]-(e2:Entity {{canonical_form: $e2_canonical, type: $e2_type}})
        )
        RETURN [node in nodes(path) | node{{.*, canonical_form:node.canonical_form, type:node.type}}] AS nodes,
               [rel in relationships(path) | rel{{.*, type:type(rel)}}] AS rels
        LIMIT 1
        """

        try:
            result = self.neo4j._run_read(query, {
                "e1_canonical": entity1.canonical_form,
                "e1_type": entity1.entity_type,
                "e2_canonical": entity2.canonical_form,
                "e2_type": entity2.entity_type,
            })

            if result:
                return {
                    "nodes": result[0]["nodes"],
                    "relationships": result[0]["rels"]
                }
        except Exception as e:
            print(f"[KGQuery] Path finding failed: {e}")

        return None

    def get_entity_subgraph(
        self,
        entities: List[Entity],
        include_neighbors: bool = True
    ) -> Dict:

        if not entities:
            return {"nodes": [], "edges": []}

        entity_filters = []
        for entity in entities:
            entity_filters.append(
                f"(e.canonical_form = '{entity.canonical_form}' AND e.type = '{entity.entity_type}')"
            )

        filter_clause = " OR ".join(entity_filters)

        if include_neighbors:
            query = f"""
            MATCH (e:Entity)
            WHERE {filter_clause}
            MATCH (e)-[r:RELATES_TO]-(neighbor:Entity)
            RETURN DISTINCT e{{.*, canonical_form:e.canonical_form, type:e.type}} AS source,
                   neighbor{{.*, canonical_form:neighbor.canonical_form, type:neighbor.type}} AS target,
                   r{{.*, rel_type:r.type}} AS relationship
            """
        else:
            query = f"""
            MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)
            WHERE ({filter_clause.replace('e.', 'e1.')}) AND ({filter_clause.replace('e.', 'e2.')})
            RETURN e1{{.*, canonical_form:e1.canonical_form, type:e1.type}} AS source,
                   e2{{.*, canonical_form:e2.canonical_form, type:e2.type}} AS target,
                   r{{.*, rel_type:r.type}} AS relationship
            """

        try:
            result = self.neo4j._run_read(query, {})

            nodes_dict = {}
            edges = []

            for record in result:
                source = record["source"]
                target = record["target"]
                rel = record["relationship"]

                source_key = (source["canonical_form"], source["type"])
                target_key = (target["canonical_form"], target["type"])

                nodes_dict[source_key] = source
                nodes_dict[target_key] = target

                edges.append({
                    "source": source["canonical_form"],
                    "target": target["canonical_form"],
                    "type": rel.get("rel_type", "RELATES_TO"),
                    "confidence": rel.get("confidence", 0.0),
                })

            return {
                "nodes": list(nodes_dict.values()),
                "edges": edges
            }

        except Exception as e:
            print(f"[KGQuery] Subgraph extraction failed: {e}")
            return {"nodes": [], "edges": []}

    def get_graph_statistics(self) -> Dict:

        stats = {}

        queries = {
            "total_entities": "MATCH (e:Entity) RETURN count(e) AS count",
            "total_relations": "MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS count",
            "total_chunks": "MATCH (c:Chunk) RETURN count(c) AS count",
            "entity_types": "MATCH (e:Entity) RETURN e.type AS type, count(e) AS count ORDER BY count DESC",
        }

        try:
            for key, query in queries.items():
                result = self.neo4j._run_read(query, {})
                if key == "entity_types":
                    stats[key] = {r["type"]: r["count"] for r in result}
                else:
                    stats[key] = result[0]["count"] if result else 0
        except Exception as e:
            print(f"[KGQuery] Failed to get statistics: {e}")

        return stats
