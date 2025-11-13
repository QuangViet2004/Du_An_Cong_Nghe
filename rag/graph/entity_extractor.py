

import re
import os
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict
import numpy as np

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

class EntityExtractor:

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
    ):
        self.mode = mode
        self.entity_types = entity_types or self.ENTITY_TYPES
        self.min_confidence = min_confidence
        self.enable_coreference = enable_coreference
        self.llm_model = llm_model
        self.text_embedder = text_embedder

        self.nlp = None
        if SPACY_AVAILABLE and mode in ["ner_only", "hybrid"]:
            try:
                self.nlp = spacy.load(spacy_model)
            except OSError:
                print(f"[EntityExtractor] spaCy model '{spacy_model}' not found. Download with: python -m spacy download {spacy_model}")
                if mode == "ner_only":
                    print("[EntityExtractor] Falling back to llm_only mode")
                    self.mode = "llm_only"

        self.groq_client = None
        if GROQ_AVAILABLE and mode in ["llm_only", "hybrid"]:
            api_key = os.getenv("GROQ_API_KEY")
            if api_key:
                self.groq_client = Groq(api_key=api_key)
            else:
                print("[EntityExtractor] GROQ_API_KEY not set. LLM extraction disabled.")
                if mode == "llm_only":
                    print("[EntityExtractor] Falling back to ner_only mode")
                    self.mode = "ner_only"

    def extract_from_chunk(self, chunk: Chunk, chunk_idx: int) -> Tuple[List[Entity], List[Relation]]:

        if not chunk.text or chunk.modality != "text":
            return [], []

        entities: List[Entity] = []
        relations: List[Relation] = []

        if self.mode == "ner_only":
            entities = self._extract_with_ner(chunk.text, chunk_idx)
        elif self.mode == "llm_only":
            entities, relations = self._extract_with_llm(chunk.text, chunk_idx)
        elif self.mode == "hybrid":

            ner_entities = self._extract_with_ner(chunk.text, chunk_idx)

            llm_entities, relations = self._extract_with_llm(chunk.text, chunk_idx, candidate_entities=ner_entities)

            entities = self._merge_entities(ner_entities, llm_entities)

        entities = [e for e in entities if e.confidence >= self.min_confidence and e.entity_type in self.entity_types]
        relations = [r for r in relations if r.confidence >= self.min_confidence]

        if self.text_embedder and entities:
            entity_texts = [e.canonical_form for e in entities]
            try:
                embeddings = self.text_embedder.encode(entity_texts)
                for i, entity in enumerate(entities):
                    entity.embedding = embeddings[i].astype(np.float32)
            except Exception as e:
                print(f"[EntityExtractor] Failed to embed entities: {e}")

        return entities, relations

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
            print(f"[EntityExtractor] LLM extraction failed: {e}")
            return [], []

    def _build_extraction_prompt(self, text: str, candidate_entities: Optional[List[Entity]] = None) -> str:

        entity_types_str = ", ".join(self.entity_types)

        prompt = f"""Extract entities and relationships from the following text.

Entity Types: {entity_types_str}

Instructions:
1. Identify all important entities and their types
2. Extract relationships between entities (subject, predicate, object)
3. Provide confidence scores (0.0-1.0) for each extraction
4. Use canonical forms for entity names (e.g., "Albert Einstein" instead of "Einstein")

Text:
\"\"\"
{text}
\"\"\"
"""

        if candidate_entities:
            candidates_str = "\n".join([f"- {e.name} ({e.entity_type})" for e in candidate_entities[:10]])
            prompt += f"\n\nCandidate Entities (from NER):\n{candidates_str}\n"

        prompt += """
Output format (JSON):
{
  "entities": [
    {"name": "original name", "canonical": "canonical form", "type": "Entity_Type", "confidence": 0.9, "description": "brief description"}
  ],
  "relations": [
    {"subject": "Entity1", "predicate": "relationship_type", "object": "Entity2", "confidence": 0.85, "evidence": "supporting text"}
  ]
}

Output:"""

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
                description = ent_data.get("description", "")

                position = original_text.lower().find(name.lower())
                if position == -1:
                    position = original_text.lower().find(canonical.lower())

                if position >= 0:
                    start = max(0, position - 50)
                    end = min(len(original_text), position + len(name) + 50)
                    context = original_text[start:end]
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
                    description=description,
                    metadata={"extraction_method": "llm"}
                )
                entities.append(entity)
                entity_map[canonical.lower()] = entity

            for rel_data in data.get("relations", []):
                subj_name = rel_data.get("subject", "")
                pred = rel_data.get("predicate", "")
                obj_name = rel_data.get("object", "")
                confidence = float(rel_data.get("confidence", 0.7))
                evidence = rel_data.get("evidence", "")

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
                    evidence=evidence,
                    metadata={"extraction_method": "llm"}
                )
                relations.append(relation)

        except Exception as e:
            print(f"[EntityExtractor] Failed to parse LLM response: {e}")
            print(f"Response: {response[:500]}")

        return entities, relations

    def _normalize_entity_name(self, name: str, entity_type: str) -> str:

        name = " ".join(name.split())

        if entity_type in ["Person", "Organization", "Location", "Product", "Event"]:

            name = name.title()
        elif entity_type == "Technical_Term":

            pass
        else:

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

    def extract_from_chunks(self, chunks: List[Chunk]) -> Tuple[List[Entity], List[Relation]]:

        all_entities: List[Entity] = []
        all_relations: List[Relation] = []

        for idx, chunk in enumerate(chunks):
            entities, relations = self.extract_from_chunk(chunk, idx)

            chunk.entities = entities
            chunk.relations = relations

            all_entities.extend(entities)
            all_relations.extend(relations)

        resolved_entities = self.resolve_entities(all_entities)

        resolved_entity_map = {
            (e.canonical_form.lower(), e.entity_type): e
            for e in resolved_entities
        }

        for relation in all_relations:
            key_subj = (relation.subject.canonical_form.lower(), relation.subject.entity_type)
            key_obj = (relation.object.canonical_form.lower(), relation.object.entity_type)

            if key_subj in resolved_entity_map:
                relation.subject = resolved_entity_map[key_subj]
            if key_obj in resolved_entity_map:
                relation.object = resolved_entity_map[key_obj]

        unique_relations = list(set(all_relations))

        return resolved_entities, unique_relations
