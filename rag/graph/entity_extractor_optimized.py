

import re
import os
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
import time

from rag.types import Entity, Relation, Chunk

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    Groq = None

class OptimizedEntityExtractor:

    ENTITY_TYPES = [
        "Person", "Organization", "Location", "Concept",
        "Date", "Number", "Technical_Term", "Product", "Event"
    ]

    SPACY_TO_ENTITY_TYPE = {
        "PERSON": "Person",
        "ORG": "Organization",
        "GPE": "Location",
        "LOC": "Location",
        "DATE": "Date",
        "TIME": "Date",
        "CARDINAL": "Number",
        "QUANTITY": "Number",
        "PRODUCT": "Product",
        "EVENT": "Event",
        "FAC": "Location",
        "NORP": "Organization",
    }

    def __init__(
        self,
        mode: str = "hybrid",
        entity_types: Optional[List[str]] = None,
        min_confidence: float = 0.7,
        enable_coreference: bool = False,
        spacy_model: str = "en_core_web_sm",
        llm_model: str = "llama-3.3-70b-versatile",
        text_embedder = None,

        batch_size: int = 10,  
        max_workers: int = 4,  
        enable_cache: bool = True,
        cache_dir: str = ".entity_cache",
        llm_threshold: int = 3,  
        max_text_length: int = 2000,  
    ):
        self.mode = mode
        self.entity_types = entity_types or self.ENTITY_TYPES
        self.min_confidence = min_confidence
        self.enable_coreference = enable_coreference
        self.llm_model = llm_model
        self.text_embedder = text_embedder

        self.batch_size = batch_size
        self.max_workers = max_workers
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir
        self.llm_threshold = llm_threshold
        self.max_text_length = max_text_length

        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "llm_calls": 0,
            "llm_skipped": 0,
        }

        if enable_cache:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file = os.path.join(cache_dir, "entity_cache.json")
            self._load_cache()
        else:
            self.cache = {}

        self.nlp = None
        if SPACY_AVAILABLE and mode in ["ner_only", "hybrid"]:
            try:
                self.nlp = spacy.load(spacy_model)

                if "parser" in self.nlp.pipe_names:
                    self.nlp.disable_pipe("parser")
                if "lemmatizer" in self.nlp.pipe_names:
                    self.nlp.disable_pipe("lemmatizer")

                print(f"[OptimizedExtractor] spaCy loaded with pipes: {self.nlp.pipe_names}")

            except OSError:
                print(f"[OptimizedExtractor] spaCy model '{spacy_model}' not found")
                if mode == "ner_only":
                    self.mode = "llm_only"

        self.groq_client = None
        if GROQ_AVAILABLE and mode in ["llm_only", "hybrid"]:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self.groq_client = Groq(api_key=api_key)
            else:
                print("[OptimizedExtractor] GROQ_API_KEY not set")
                if mode == "llm_only":
                    self.mode = "ner_only"

    def _load_cache(self):

        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                print(f"[OptimizedExtractor] Loaded cache with {len(self.cache)} entries")
            except Exception as e:
                print(f"[OptimizedExtractor] Cache load failed: {e}")
                self.cache = {}
        else:
            self.cache = {}

    def _save_cache(self):

        if self.enable_cache:
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.cache, f, indent=2)
            except Exception as e:
                print(f"[OptimizedExtractor] Cache save failed: {e}")

    def _get_cache_key(self, text: str) -> str:

        return hashlib.md5(text.encode()).hexdigest()

    def print_stats(self):

        total = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total > 0:
            hit_rate = self.stats["cache_hits"] / total * 100
            print(f"\n[OptimizedExtractor] Performance Stats:")
            print(f"  Cache hits: {self.stats['cache_hits']}/{total} ({hit_rate:.1f}%)")
            print(f"  LLM calls: {self.stats['llm_calls']}")
            print(f"  LLM skipped (smart filtering): {self.stats['llm_skipped']}")

    def extract_from_chunks(self, chunks: List[Chunk]) -> Tuple[List[Entity], List[Relation]]:

        start_time = time.time()

        print(f"[OptimizedExtractor] Processing {len(chunks)} chunks in parallel (workers={self.max_workers})...")

        all_entities: List[Entity] = []
        all_relations: List[Relation] = []

        text_chunks = [(idx, chunk) for idx, chunk in enumerate(chunks) if chunk.text and chunk.modality == "text"]

        if not text_chunks:
            return [], []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.extract_from_chunk, chunk, idx): (idx, chunk)
                for idx, chunk in text_chunks
            }

            processed = 0
            for future in as_completed(futures):
                idx, chunk = futures[future]
                try:
                    entities, relations = future.result()
                    chunk.entities = entities
                    chunk.relations = relations
                    all_entities.extend(entities)
                    all_relations.extend(relations)

                    processed += 1
                    if processed % 10 == 0:
                        print(f"[OptimizedExtractor] Processed {processed}/{len(text_chunks)} chunks...")

                except Exception as e:
                    print(f"[OptimizedExtractor] Failed to process chunk {idx}: {e}")

        if all_entities:
            self.embed_entities_batch(all_entities)

        print(f"[OptimizedExtractor] Resolving {len(all_entities)} entities...")
        resolved_entities = self.resolve_entities(all_entities)

        entity_map = {
            (e.canonical_form.lower(), e.entity_type): e
            for e in resolved_entities
        }

        for relation in all_relations:
            key_subj = (relation.subject.canonical_form.lower(), relation.subject.entity_type)
            key_obj = (relation.object.canonical_form.lower(), relation.object.entity_type)
            if key_subj in entity_map:
                relation.subject = entity_map[key_subj]
            if key_obj in entity_map:
                relation.object = entity_map[key_obj]

        unique_relations = list(set(all_relations))

        self._save_cache()

        elapsed = time.time() - start_time
        print(f"[OptimizedExtractor] Complete in {elapsed:.1f}s: {len(resolved_entities)} entities, {len(unique_relations)} relations")
        self.print_stats()

        return resolved_entities, unique_relations

    def extract_from_chunk(self, chunk: Chunk, chunk_idx: int) -> Tuple[List[Entity], List[Relation]]:

        if not chunk.text or chunk.modality != "text":
            return [], []

        cache_key = self._get_cache_key(chunk.text)
        if self.enable_cache and cache_key in self.cache:
            self.stats["cache_hits"] += 1
            cached = self.cache[cache_key]
            entities = self._deserialize_entities(cached.get("entities", []), chunk_idx)
            relations = self._deserialize_relations(cached.get("relations", []), entities, chunk_idx)
            return entities, relations

        self.stats["cache_misses"] += 1

        entities: List[Entity] = []
        relations: List[Relation] = []

        if self.mode == "ner_only":
            entities = self._extract_with_ner(chunk.text, chunk_idx)

        elif self.mode == "llm_only":
            entities, relations = self._extract_with_llm(chunk.text, chunk_idx)

        elif self.mode == "hybrid":

            ner_entities = self._extract_with_ner(chunk.text, chunk_idx)

            if len(ner_entities) >= self.llm_threshold:
                self.stats["llm_calls"] += 1
                llm_entities, relations = self._extract_with_llm(chunk.text, chunk_idx, ner_entities)
                entities = self._merge_entities(ner_entities, llm_entities)
            else:

                self.stats["llm_skipped"] += 1
                entities = ner_entities

        entities = [e for e in entities if e.confidence >= self.min_confidence and e.entity_type in self.entity_types]
        relations = [r for r in relations if r.confidence >= self.min_confidence]

        if self.enable_cache:
            self.cache[cache_key] = {
                "entities": self._serialize_entities(entities),
                "relations": self._serialize_relations(relations),
            }

        return entities, relations

    def embed_entities_batch(self, entities: List[Entity]) -> None:

        if not self.text_embedder or not entities:
            return

        print(f"[OptimizedExtractor] Batch embedding {len(entities)} entities...")
        start_time = time.time()

        entity_texts = [e.canonical_form for e in entities]

        try:

            embeddings = self.text_embedder.encode(entity_texts)

            for i, entity in enumerate(entities):
                entity.embedding = embeddings[i].astype(np.float32)

            elapsed = time.time() - start_time
            print(f"[OptimizedExtractor] Batch embedding done in {elapsed:.2f}s")

        except Exception as e:
            print(f"[OptimizedExtractor] Batch embedding failed: {e}")

    def _extract_with_ner(self, text: str, chunk_idx: int) -> List[Entity]:

        if not self.nlp:
            return []

        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entity_type = self.SPACY_TO_ENTITY_TYPE.get(ent.label_, "Concept")

            if entity_type not in self.entity_types:
                continue

            canonical = self._normalize_entity_name(ent.text, entity_type)

            start = max(0, ent.start_char - 50)
            end = min(len(text), ent.end_char + 50)
            context = text[start:end]

            entity = Entity(
                name=ent.text,
                canonical_form=canonical,
                entity_type=entity_type,
                confidence=0.8,
                chunk_idx=chunk_idx,
                position=ent.start_char,
                context=context,
                metadata={"spacy_label": ent.label_}
            )
            entities.append(entity)

        return entities

    def _extract_with_llm(
        self,
        text: str,
        chunk_idx: int,
        candidate_entities: Optional[List[Entity]] = None
    ) -> Tuple[List[Entity], List[Relation]]:

        if not self.groq_client:
            return [], []

        if len(text) > self.max_text_length:
            text = text[:self.max_text_length] + "..."

        prompt = self._build_extraction_prompt(text, candidate_entities)

        try:
            response = self.groq_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,
            )

            result = response.choices[0].message.content
            entities, relations = self._parse_llm_response(result, chunk_idx, text)
            return entities, relations

        except Exception as e:
            print(f"[OptimizedExtractor] LLM call failed: {e}")
            return [], []

    def _build_extraction_prompt(self, text: str, candidate_entities: Optional[List[Entity]] = None) -> str:

        entity_types_str = ", ".join(self.entity_types)

        prompt = f"""Extract key entities and relations.

Types: {entity_types_str}

Text:
\"\"\"
{text}
\"\"\"
"""

        if candidate_entities:

            candidates = [f"{e.name} ({e.entity_type})" for e in candidate_entities[:5]]
            prompt += f"\nKnown: {', '.join(candidates)}\n"

        prompt += """
Output JSON:
{"entities": [{"name": "...", "canonical": "...", "type": "...", "confidence": 0.9}],
 "relations": [{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.8}]}
"""

        return prompt

    def _parse_llm_response(self, response: str, chunk_idx: int, original_text: str) -> Tuple[List[Entity], List[Relation]]:

        import json

        entities: List[Entity] = []
        relations: List[Relation] = []

        try:

            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                json_str = json_match.group(0) if json_match else response

            data = json.loads(json_str)

            entity_map: Dict[str, Entity] = {}

            for ent_data in data.get("entities", []):
                name = ent_data.get("name", "")
                canonical = ent_data.get("canonical", name)
                entity_type = ent_data.get("type", "Concept")
                confidence = float(ent_data.get("confidence", 0.7))

                position = original_text.lower().find(name.lower())

                if position >= 0:
                    context = original_text[max(0, position-50):position+len(name)+50]
                else:
                    context = original_text[:100]

                entity = Entity(
                    name=name,
                    canonical_form=canonical,
                    entity_type=entity_type,
                    confidence=confidence,
                    chunk_idx=chunk_idx,
                    position=position if position >= 0 else None,
                    context=context,
                    description=ent_data.get("description"),
                    metadata={"extraction_method": "llm"}
                )
                entities.append(entity)
                entity_map[canonical.lower()] = entity

            for rel_data in data.get("relations", []):
                subj_name = rel_data.get("subject", "")
                pred = rel_data.get("predicate", "")
                obj_name = rel_data.get("object", "")
                confidence = float(rel_data.get("confidence", 0.7))

                subj = entity_map.get(subj_name.lower())
                obj = entity_map.get(obj_name.lower())

                if not subj:
                    subj = Entity(
                        name=subj_name,
                        canonical_form=subj_name,
                        entity_type="Concept",
                        confidence=0.6,
                        chunk_idx=chunk_idx
                    )
                    entities.append(subj)

                if not obj:
                    obj = Entity(
                        name=obj_name,
                        canonical_form=obj_name,
                        entity_type="Concept",
                        confidence=0.6,
                        chunk_idx=chunk_idx
                    )
                    entities.append(obj)

                relation = Relation(
                    subject=subj,
                    predicate=pred,
                    object=obj,
                    confidence=confidence,
                    chunk_idx=chunk_idx,
                    evidence=rel_data.get("evidence"),
                    metadata={"extraction_method": "llm"}
                )
                relations.append(relation)

        except Exception as e:
            print(f"[OptimizedExtractor] Parse error: {e}")

        return entities, relations

    def _normalize_entity_name(self, name: str, entity_type: str) -> str:

        name = " ".join(name.split())

        if entity_type in ["Person", "Organization", "Location", "Product", "Event"]:
            name = name.title()
        elif entity_type != "Technical_Term":
            name = name.lower()

        return name

    def _merge_entities(self, ner_entities: List[Entity], llm_entities: List[Entity]) -> List[Entity]:

        entity_map: Dict[Tuple[str, str], Entity] = {}

        for entity in ner_entities:
            key = (entity.canonical_form.lower(), entity.entity_type)
            entity_map[key] = entity

        for entity in llm_entities:
            key = (entity.canonical_form.lower(), entity.entity_type)
            if key in entity_map:
                existing = entity_map[key]
                if existing.position is not None and entity.position is None:
                    entity.position = existing.position
                if existing.context and not entity.context:
                    entity.context = existing.context
            entity_map[key] = entity

        return list(entity_map.values())

    def resolve_entities(self, all_entities: List[Entity]) -> List[Entity]:

        entity_groups: Dict[str, List[Entity]] = defaultdict(list)

        for entity in all_entities:
            key = entity.canonical_form.lower()
            entity_groups[key].append(entity)

        resolved: List[Entity] = []

        for canonical_key, group in entity_groups.items():
            if len(group) == 1:
                resolved.append(group[0])
                continue

            group.sort(key=lambda e: e.confidence, reverse=True)
            primary = group[0]

            primary.metadata["occurrences"] = len(group)
            primary.metadata["chunk_indices"] = [e.chunk_idx for e in group if e.chunk_idx is not None]

            for entity in group:
                if entity.description and not primary.description:
                    primary.description = entity.description
                    break

            resolved.append(primary)

        return resolved

    def _serialize_entities(self, entities: List[Entity]) -> List[Dict]:

        return [
            {
                "name": e.name,
                "canonical_form": e.canonical_form,
                "entity_type": e.entity_type,
                "confidence": e.confidence,
                "position": e.position,
                "description": e.description,
            }
            for e in entities
        ]

    def _deserialize_entities(self, data: List[Dict], chunk_idx: int) -> List[Entity]:

        return [
            Entity(
                name=d["name"],
                canonical_form=d["canonical_form"],
                entity_type=d["entity_type"],
                confidence=d["confidence"],
                chunk_idx=chunk_idx,
                position=d.get("position"),
                description=d.get("description"),
            )
            for d in data
        ]

    def _serialize_relations(self, relations: List[Relation]) -> List[Dict]:

        return [
            {
                "subject": r.subject.canonical_form,
                "subject_type": r.subject.entity_type,
                "predicate": r.predicate,
                "object": r.object.canonical_form,
                "object_type": r.object.entity_type,
                "confidence": r.confidence,
                "evidence": r.evidence,
            }
            for r in relations
        ]

    def _deserialize_relations(self, data: List[Dict], entities: List[Entity], chunk_idx: int) -> List[Relation]:

        entity_map = {(e.canonical_form, e.entity_type): e for e in entities}

        relations = []
        for d in data:
            subj = entity_map.get((d["subject"], d.get("subject_type", "Concept")))
            obj = entity_map.get((d["object"], d.get("object_type", "Concept")))

            if not subj:
                subj = Entity(
                    name=d["subject"],
                    canonical_form=d["subject"],
                    entity_type=d.get("subject_type", "Concept"),
                    confidence=0.5,
                    chunk_idx=chunk_idx
                )
            if not obj:
                obj = Entity(
                    name=d["object"],
                    canonical_form=d["object"],
                    entity_type=d.get("object_type", "Concept"),
                    confidence=0.5,
                    chunk_idx=chunk_idx
                )

            relation = Relation(
                subject=subj,
                predicate=d["predicate"],
                object=obj,
                confidence=d["confidence"],
                chunk_idx=chunk_idx,
                evidence=d.get("evidence"),
            )
            relations.append(relation)

        return relations
