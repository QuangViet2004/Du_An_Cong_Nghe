from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from neo4j import GraphDatabase
except Exception:
    GraphDatabase = None  

class Neo4jGraphStore:
    def __init__(self, uri: str, user: str, password: str, database: Optional[str] = None):
        if GraphDatabase is None:
            raise RuntimeError("neo4j driver not available")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def _run(self, query: str, params: Dict):
        with self.driver.session(database=self.database) as session:
            return session.run(query, params)

    def _run_read(self, query: str, params: Dict):

        with self.driver.session(database=self.database) as session:
            result = session.run(query, params)
            return list(result)

    def ensure_indexes(self, dim: int):

        try:
            self._run("CREATE CONSTRAINT doc_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.name IS UNIQUE", {})
            self._run("CREATE INDEX chunk_text_idx IF NOT EXISTS FOR (c:Chunk) ON (c.text)", {})
        except Exception:
            pass
        try:
            self._run(
                "CALL db.index.vector.createNodeIndex('chunk_embedding_index','Chunk','embedding',$dim,'cosine')",
                {"dim": dim},
            )
        except Exception:
            pass

    def ensure_entity_indexes(self):

        try:

            self._run(
                "CREATE CONSTRAINT entity_unique IF NOT EXISTS FOR (e:Entity) REQUIRE (e.canonical_form, e.type) IS UNIQUE",
                {}
            )
            self._run("CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)", {})
            self._run("CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)", {})

            try:
                self._run(
                    "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (e:Entity) ON EACH [e.name, e.canonical_form, e.description]",
                    {}
                )
            except Exception:
                pass

            try:

                result = self._run_read(
                    "MATCH (e:Entity) WHERE e.embedding IS NOT NULL RETURN e.embedding LIMIT 1",
                    {}
                )
                if result:
                    embedding = result[0]["e.embedding"]
                    dim = len(embedding)
                    self._run(
                        "CALL db.index.vector.createNodeIndex('entity_embedding_index','Entity','embedding',$dim,'cosine')",
                        {"dim": dim}
                    )
            except Exception:
                pass

            self._run("CREATE INDEX rel_type_idx IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.type)", {})
            self._run("CREATE INDEX mentions_chunk_idx IF NOT EXISTS FOR ()-[r:MENTIONS]-() ON (r.chunk_idx)", {})

            print("[Neo4j] Entity indexes created successfully")

        except Exception as e:
            print(f"[Neo4j] Failed to create entity indexes: {e}")

    def upsert_rows(self, rows: List[Dict]):

        if not rows:
            return
        source = rows[0].get("source", "dataset")
        query = (
            "MERGE (d:Document {name:$source}) "
            "WITH d, $rows AS rows "
            "UNWIND rows AS r "
            "MERGE (p:Page {number:r.page, source:r.source}) "
            "MERGE (d)-[:HAS_PAGE]->(p) "
            "MERGE (c:Chunk {source:r.source, page:r.page, idx:r.idx}) "
            "SET c.text = r.text, c.heading = r.heading, c.embedding = r.embedding, c.image_path = r.image_path, c.tags = r.tags "
            "MERGE (p)-[:HAS_CHUNK]->(c)"
        )
        self._run(query, {"source": source, "rows": rows})

    def upsert_entities(self, entities: List) -> None:

        if not entities:
            return

        entity_dicts = []
        for entity in entities:
            entity_dict = {
                "name": entity.name,
                "canonical_form": entity.canonical_form,
                "type": entity.entity_type,
                "confidence": entity.confidence,
                "description": entity.description or "",
                "chunk_idx": entity.chunk_idx,
                "position": entity.position,
                "context": entity.context or "",
            }

            if entity.embedding is not None:
                if isinstance(entity.embedding, np.ndarray):
                    entity_dict["embedding"] = entity.embedding.tolist()
                else:
                    entity_dict["embedding"] = entity.embedding

            if entity.metadata:
                entity_dict["metadata"] = str(entity.metadata)  
                if "importance_score" in entity.metadata:
                    entity_dict["importance_score"] = entity.metadata["importance_score"]
                if "occurrences" in entity.metadata:
                    entity_dict["occurrences"] = entity.metadata["occurrences"]

            entity_dicts.append(entity_dict)

        query = """
        UNWIND $entities AS e
        MERGE (entity:Entity {canonical_form: e.canonical_form, type: e.type})
        SET entity.name = e.name,
            entity.confidence = e.confidence,
            entity.description = e.description,
            entity.context = e.context,
            entity.metadata = e.metadata,
            entity.importance_score = COALESCE(e.importance_score, entity.importance_score),
            entity.occurrences = COALESCE(e.occurrences, entity.occurrences)
        WITH entity, e
        WHERE e.embedding IS NOT NULL
        SET entity.embedding = e.embedding
        """

        try:
            self._run(query, {"entities": entity_dicts})
            print(f"[Neo4j] Upserted {len(entity_dicts)} entities")
        except Exception as e:
            print(f"[Neo4j] Failed to upsert entities: {e}")

    def upsert_relations(self, relations: List) -> None:

        if not relations:
            return

        relation_dicts = []
        for relation in relations:
            relation_dict = {
                "subject_canonical": relation.subject.canonical_form,
                "subject_type": relation.subject.entity_type,
                "predicate": relation.predicate,
                "object_canonical": relation.object.canonical_form,
                "object_type": relation.object.entity_type,
                "confidence": relation.confidence,
                "chunk_idx": relation.chunk_idx,
                "evidence": relation.evidence or "",
            }
            relation_dicts.append(relation_dict)

        query = """
        UNWIND $relations AS r
        MATCH (subj:Entity {canonical_form: r.subject_canonical, type: r.subject_type})
        MATCH (obj:Entity {canonical_form: r.object_canonical, type: r.object_type})
        MERGE (subj)-[rel:RELATES_TO {type: r.predicate}]->(obj)
        SET rel.confidence = r.confidence,
            rel.chunk_idx = r.chunk_idx,
            rel.evidence = r.evidence
        """

        try:
            self._run(query, {"relations": relation_dicts})
            print(f"[Neo4j] Upserted {len(relation_dicts)} relations")
        except Exception as e:
            print(f"[Neo4j] Failed to upsert relations: {e}")

    def link_entities_to_chunks(self, chunks: List) -> None:

        mention_dicts = []

        for chunk in chunks:
            if not chunk.entities:
                continue

            for entity in chunk.entities:
                mention_dict = {
                    "chunk_source": chunk.source,
                    "chunk_page": chunk.page,
                    "chunk_idx": getattr(chunk, "idx", 0),  
                    "entity_canonical": entity.canonical_form,
                    "entity_type": entity.entity_type,
                    "position": entity.position or 0,
                    "context": entity.context or "",
                }
                mention_dicts.append(mention_dict)

        if not mention_dicts:
            return

        query = """
        UNWIND $mentions AS m
        MATCH (c:Chunk {source: m.chunk_source, page: m.chunk_page})
        MATCH (e:Entity {canonical_form: m.entity_canonical, type: m.entity_type})
        MERGE (c)-[rel:MENTIONS]->(e)
        SET rel.position = m.position,
            rel.context = m.context
        """

        try:
            self._run(query, {"mentions": mention_dicts})
            print(f"[Neo4j] Created {len(mention_dicts)} MENTIONS relationships")
        except Exception as e:
            print(f"[Neo4j] Failed to link entities to chunks: {e}")

    def vector_search(self, query_vec, top_k: int = 5):

        try:
            res = self._run(
                "CALL db.index.vector.queryNodes('chunk_embedding_index', $k, $v) YIELD node, score RETURN node{.*, source:node.source, page:node.page, idx:node.idx} AS node, score LIMIT $k",
                {"k": top_k, "v": query_vec.tolist()},
            )
            out = []
            for r in res:
                node = r["node"]
                out.append((node, float(r["score"])))
            return out
        except Exception:
            return []

    def entity_vector_search(self, query_vec, top_k: int = 10) -> List[Tuple[Dict, float]]:

        try:
            res = self._run(
                "CALL db.index.vector.queryNodes('entity_embedding_index', $k, $v) YIELD node, score RETURN node{.*, canonical_form:node.canonical_form, type:node.type, name:node.name} AS node, score LIMIT $k",
                {"k": top_k, "v": query_vec.tolist()},
            )
            out = []
            for r in res:
                node = r["node"]
                out.append((node, float(r["score"])))
            return out
        except Exception as e:
            print(f"[Neo4j] Entity vector search failed: {e}")
            return []

    def get_entity_neighbors(self, entity_canonical: str, entity_type: str, hops: int = 1) -> List[Dict]:

        hops = min(max(hops, 1), 3)  

        query = f"""
        MATCH (e:Entity {{canonical_form: $canonical, type: $type}})
        MATCH path = (e)-[:RELATES_TO*1..{hops}]-(neighbor:Entity)
        RETURN DISTINCT neighbor{{.*, canonical_form:neighbor.canonical_form, type:neighbor.type, name:neighbor.name}} AS neighbor,
               length(path) AS distance
        ORDER BY distance, neighbor.importance_score DESC
        LIMIT 50
        """

        try:
            result = self._run_read(query, {"canonical": entity_canonical, "type": entity_type})
            neighbors = [r["neighbor"] for r in result]
            return neighbors
        except Exception as e:
            print(f"[Neo4j] Failed to get entity neighbors: {e}")
            return []

    def get_entity_chunks(self, entity_canonical: str, entity_type: str, limit: int = 10) -> List[Dict]:

        query = """
        MATCH (e:Entity {canonical_form: $canonical, type: $type})<-[:MENTIONS]-(c:Chunk)
        RETURN c{.*, source:c.source, page:c.page, idx:c.idx, text:c.text} AS chunk
        ORDER BY c.importance_score DESC
        LIMIT $limit
        """

        try:
            result = self._run_read(query, {"canonical": entity_canonical, "type": entity_type, "limit": limit})
            chunks = [r["chunk"] for r in result]
            return chunks
        except Exception as e:
            print(f"[Neo4j] Failed to get entity chunks: {e}")
            return []

    def find_entities_by_name(self, name_query: str, limit: int = 10) -> List[Dict]:

        query = """
        CALL db.index.fulltext.queryNodes('entity_fulltext', $query) YIELD node, score
        RETURN node{.*, canonical_form:node.canonical_form, type:node.type, name:node.name} AS entity, score
        ORDER BY score DESC
        LIMIT $limit
        """

        try:
            result = self._run_read(query, {"query": name_query, "limit": limit})
            entities = [(r["entity"], r["score"]) for r in result]
            return entities
        except Exception as e:
            print(f"[Neo4j] Full-text search failed: {e}")

            fallback_query = """
            MATCH (e:Entity)
            WHERE e.name CONTAINS $query OR e.canonical_form CONTAINS $query
            RETURN e{.*, canonical_form:e.canonical_form, type:e.type, name:e.name} AS entity
            LIMIT $limit
            """
            try:
                result = self._run_read(fallback_query, {"query": name_query, "limit": limit})
                entities = [(r["entity"], 1.0) for r in result]
                return entities
            except Exception:
                return []
