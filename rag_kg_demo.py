

import os
from typing import List

from rag.types import Chunk
from rag.loaders.doc_loader import load_pdf_with_docling
from rag.chunking.semantic_chunker import section_chunk_blocks
from rag.embeddings.text_embedder import TextEmbedder
from rag.retrieval.hybrid_indexer import HybridIndexer
from rag.graph.neo4j_store import Neo4jGraphStore
from rag.graph.entity_extractor import EntityExtractor
from rag.graph.kg_builder import KnowledgeGraphBuilder
from rag.retrieval.kg_enhanced_retriever import KGEnhancedRetriever
from rag.prompts.kg_citation_prompt import build_prompt_with_kg_context
from rag.llm.groq_client import groq_stream_completion
from rag.config.kg_config import get_kg_config, PRESET_MODES
from rag.cache.cache_io import (
    compute_dataset_md5,
    load_cache,
    save_cache,
)

def main():

    pdf_files = [
        r"1512.03385v1.pdf",  
    ]

    kg_mode = "balanced"  
    kg_config = get_kg_config(kg_mode)

    target_chars = 900
    overlap_chars = 150
    top_k = 5
    alpha = 0.6
    text_model = "Alibaba-NLP/gte-multilingual-base"
    vlm_caption_model = "ibm-granite/granite-docling-258M"

    use_neo4j = True
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

    dataset_key = compute_dataset_md5(pdf_files)
    cache_dir = os.path.join(os.getcwd(), ".rag_cache")
    image_output_dir = os.path.join(cache_dir, dataset_key, "images")
    os.makedirs(image_output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"RAG with Knowledge Graph Pipeline - Mode: {kg_mode.upper()}")
    print(f"{'='*60}\n")

    text_embedder = TextEmbedder(model_name=text_model)

    neo4j_store = None
    if use_neo4j and kg_config.store_in_neo4j:
        neo4j_store = Neo4jGraphStore(neo4j_uri, neo4j_user, neo4j_password)
        print("[Neo4j] Connected to graph database")

    cached_chunks, cached_emb, cached_meta, cached_graph = load_cache(cache_dir, dataset_key)

    kg_cache_path = os.path.join(cache_dir, dataset_key, "kg_data.json")
    cached_kg_data = None
    if os.path.exists(kg_cache_path):
        import json
        with open(kg_cache_path, 'r') as f:
            cached_kg_data = json.load(f)
            print(f"[Cache] Loaded KG data: {len(cached_kg_data.get('entities', []))} entities")

    indexer = HybridIndexer(
        text_model_name=text_model,
        image_caption_model=vlm_caption_model,
        use_vlm_caption=True,
        neo4j_store=neo4j_store
    )

    if cached_chunks is not None and cached_emb is not None and not kg_config.enabled:

        print("[Cache] Using cached chunks and embeddings (no KG extraction)")
        indexer.build_with_embeddings(cached_chunks, cached_emb)
        fine_chunks = cached_chunks

    else:

        print(f"[Load] Processing {len(pdf_files)} PDF(s)...")
        raw_chunks: List[Chunk] = []

        for pdf in pdf_files:
            print(f"  - {pdf}")
            raw_chunks.extend(load_pdf_with_docling(pdf, image_out_dir=image_output_dir))

        n_text = sum(1 for c in raw_chunks if c.modality == "text" and c.text)
        n_img = sum(1 for c in raw_chunks if c.modality == "image")
        print(f"[Load] Extracted: {n_text} text chunks, {n_img} images\n")

        print("[Chunk] Section-based chunking...")
        fine_chunks = section_chunk_blocks(raw_chunks, tag_keywords=[])
        print(f"[Chunk] Created {len(fine_chunks)} fine-grained chunks\n")

        print("[Index] Building hybrid index...")
        indexer.build(fine_chunks)
        print()

        entities = []
        relations = []

        if kg_config.enabled:
            print(f"[KG] Starting entity extraction (mode: {kg_config.extraction.mode})...")

            entity_extractor = EntityExtractor(
                mode=kg_config.extraction.mode,
                entity_types=kg_config.extraction.entity_types,
                min_confidence=kg_config.extraction.min_confidence,
                enable_coreference=kg_config.extraction.enable_coreference,
                llm_model=kg_config.extraction.llm_model,
                text_embedder=text_embedder,
            )

            kg_builder = KnowledgeGraphBuilder(
                entity_extractor=entity_extractor,
                neo4j_store=neo4j_store if kg_config.store_in_neo4j else None,
                enable_enrichment=kg_config.enable_enrichment,
            )

            kg_result = kg_builder.build(fine_chunks)

            entities = kg_result["entities"]
            relations = kg_result["relations"]

            print(f"\n[KG] Extraction complete:")
            print(f"  - Entities: {len(entities)}")
            print(f"  - Relations: {len(relations)}")

            if entities:
                print("\n[KG] Sample entities:")
                for entity in entities[:5]:
                    print(f"  - {entity.canonical_form} ({entity.entity_type}) [confidence: {entity.confidence:.2f}]")

            import json
            kg_data = {
                "entities": [
                    {
                        "name": e.name,
                        "canonical_form": e.canonical_form,
                        "type": e.entity_type,
                        "confidence": e.confidence,
                        "description": e.description,
                    }
                    for e in entities
                ],
                "relations": [
                    {
                        "subject": r.subject.canonical_form,
                        "predicate": r.predicate,
                        "object": r.object.canonical_form,
                        "confidence": r.confidence,
                    }
                    for r in relations
                ],
            }
            with open(kg_cache_path, 'w') as f:
                json.dump(kg_data, f, indent=2)

        meta = {
            "text_model": text_model,
            "image_caption_model": vlm_caption_model,
            "chunker": "section",
            "kg_enabled": kg_config.enabled,
            "kg_mode": kg_mode,
        }

        graph_rows = []
        if indexer.emb_matrix is not None:
            for idx, ch in enumerate(fine_chunks):
                graph_rows.append({
                    "source": ch.source,
                    "page": ch.page,
                    "heading": ch.heading or "",
                    "idx": idx,
                    "text": ch.text,
                    "image_path": getattr(ch, "image_path", None),
                    "tags": getattr(ch, "tags", None),
                    "embedding": indexer.emb_matrix[idx].tolist(),
                })

        save_cache(cache_dir, dataset_key, fine_chunks, indexer.emb_matrix, meta, graph_rows)
        print(f"\n[Cache] Saved to {cache_dir}/{dataset_key}\n")

    query = "What are the main components described in the document?"

    print(f"{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")

    if kg_config.enabled and kg_config.retrieval.retrieval_mode != "baseline":

        print(f"[Retrieval] Using KG-enhanced retrieval (mode: {kg_config.retrieval.retrieval_mode})")

        kg_retriever = KGEnhancedRetriever(
            base_indexer=indexer,
            neo4j_store=neo4j_store,
            text_embedder=text_embedder,
            retrieval_mode=kg_config.retrieval.retrieval_mode,
            expansion_hops=kg_config.retrieval.expansion_hops,
            entity_boost_weight=kg_config.retrieval.entity_boost_weight,
            centrality_weight=kg_config.retrieval.centrality_weight,
        )

        hits = kg_retriever.search(query, top_k=top_k, alpha=alpha)

        entity_context = None
        if kg_config.retrieval.include_entity_context:
            entity_context = kg_retriever.get_entity_context_for_results(hits)

    else:

        print("[Retrieval] Using baseline hybrid retrieval")
        hits = indexer.search(query, top_k=top_k, alpha=alpha)
        entity_context = None

    print(f"\n[Results] Top {len(hits)} chunks:\n")
    for i, (chunk, combo, ss, bs) in enumerate(hits, 1):
        print(f"{i}. Score={combo:.3f} (sem={ss:.3f}, bm25={bs:.3f})")
        print(f"   Source: {chunk.source}, Page: {chunk.page}, Heading: {chunk.heading or 'N/A'}")

        if chunk.entities:
            entity_names = [e.canonical_form for e in chunk.entities[:3]]
            print(f"   Entities: {', '.join(entity_names)}")

        preview = chunk.text[:150].replace("\n", " ")
        print(f"   Text: {preview}...")
        print()

    if kg_config.enabled and entity_context:
        prompt = build_prompt_with_kg_context(
            query=query,
            hits=hits,
            entity_context=entity_context,
            include_relationships=True,
            include_graph_viz=False,
        )
    else:
        from rag.prompts.kg_citation_prompt import build_prompt_with_citations
        prompt = build_prompt_with_citations(query, hits)

    print(f"{'='*60}")
    print("LLM Answer:")
    print(f"{'='*60}\n")

    groq_stream_completion(prompt)
    print("\n")

    if neo4j_store:
        neo4j_store.close()
        print("[Neo4j] Connection closed")

if __name__ == "__main__":
    main()
