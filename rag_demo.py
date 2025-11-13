

import os
from typing import List, Tuple, Optional
import numpy as np

from rag.types import Chunk
from rag.loaders.doc_loader import load_pdf_with_docling
from rag.chunking.semantic_chunker import (
    semantic_chunk_blocks,
    semantic_chunk_blocks_docling,
    section_chunk_blocks,
)
from rag.embeddings.text_embedder import TextEmbedder
from rag.retrieval.hybrid_indexer import HybridIndexer
from rag.graph.neo4j_store import Neo4jGraphStore
from rag.llm.groq_client import groq_stream_completion
from rag.cache.cache_io import (
    compute_dataset_md5,
    load_cache,
    save_cache,
)

def build_prompt_with_citations(query: str, hits: List[Tuple[Chunk, float, float, float]]) -> str:
    header = (
        "You are a retrieval QA system. Use ONLY the provided contexts to answer.\n"
        "For every factual statement, add an inline citation in the form "
        "[file:{source}, page:{page}, heading:{heading}] and include image link when available as [image:{image_path}].\n"
        "If something is not in the contexts, say you don't know.\n\n"
    )
    ctx = []
    for i, (chunk, combo, sem, bm) in enumerate(hits, 1):
        heading = chunk.heading if chunk.heading else "N/A"
        block = (
            f"### Context {i}\n"
            f"Source: file={chunk.source}, page={chunk.page}, heading={heading}\n"
        )
        if getattr(chunk, "image_path", None):
            block += f"Image: {chunk.image_path}\n"
        block += f"{chunk.text}\n"
        ctx.append(block)
    context_block = "\n".join(ctx)
    ask = f"\n### Question\n{query}\n\n### Answer (with citations):\n"
    return header + context_block + ask

def main():

    pdf_files = [
        r"1512.03385v1.pdf",
    ]
    target_chars = 900
    overlap_chars = 150
    top_k = 5
    alpha = 0.6
    docling_text_model = "Alibaba-NLP/gte-multilingual-base"

    vlm_caption_model = "ibm-granite/granite-docling-258M"
    use_docling_chunker = True
    use_section_chunker = True
    section_tag_keywords: List[str] = []  
    use_neo4j = True
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

    dataset_key = compute_dataset_md5(pdf_files)
    cache_dir = os.path.join(os.getcwd(), ".rag_cache")

    image_output_dir = os.path.join(cache_dir, dataset_key, "images")
    os.makedirs(image_output_dir, exist_ok=True)

    cached_chunks, cached_emb, cached_meta, cached_graph = load_cache(cache_dir, dataset_key)
    if cached_meta is not None:
        if cached_meta.get("text_model") != docling_text_model or cached_meta.get("image_caption_model") != vlm_caption_model:
            cached_chunks, cached_emb = None, None
        desired_chunker = "section" if use_section_chunker else ("docling_semantic" if use_docling_chunker else "baseline")
        if cached_meta.get("chunker") != desired_chunker:
            cached_chunks, cached_emb = None, None

        if cached_meta.get("images_saved") != "figures_only":
            cached_chunks, cached_emb = None, None

    neo4j_store: Optional[Neo4jGraphStore] = None
    if use_neo4j:
        neo4j_store = Neo4jGraphStore(neo4j_uri, neo4j_user, neo4j_password)

    indexer = HybridIndexer(text_model_name=docling_text_model, image_caption_model=vlm_caption_model, use_vlm_caption=True, neo4j_store=neo4j_store)

    if cached_chunks is not None and cached_emb is not None:

        print("[Cache] Using cached chunks and embeddings; skipping rebuild and upsert")
        indexer.build_with_embeddings(cached_chunks, cached_emb)
        fine_chunks = cached_chunks
    else:

        raw_chunks: List[Chunk] = []
        for pdf in pdf_files:
            print(f"[Load] Parsing PDF via Docling/fallback: {pdf}")
            raw_chunks.extend(load_pdf_with_docling(pdf, image_out_dir=image_output_dir))
        n_text = sum(1 for c in raw_chunks if c.modality == "text" and c.text)
        n_img = sum(1 for c in raw_chunks if c.modality == "image")
        print(f"[Load] Collected chunks -> text: {n_text}, image: {n_img}")

        text_embedder = TextEmbedder(model_name=docling_text_model)
        if use_section_chunker:
            print("[Chunk] Section-based chunking (title/content/tags)")
            fine_chunks = section_chunk_blocks(raw_chunks, tag_keywords=section_tag_keywords)
        elif use_docling_chunker:
            print("[Chunk] Semantic chunking (Docling embeddings)")
            fine_chunks = semantic_chunk_blocks_docling(
                raw_chunks,
                encode_sentences=lambda sents: text_embedder.encode(sents),
                target_chars=target_chars,
                overlap_chars=overlap_chars,
            )
        else:
            print("[Chunk] Baseline semantic chunking by size")
            fine_chunks = semantic_chunk_blocks(raw_chunks, target_chars=target_chars, overlap_chars=overlap_chars)

        print("[Index] Building hybrid index (text embed + BM25 + image captions)")
        indexer.build(fine_chunks)

        meta = {
            "text_model": docling_text_model,
            "image_caption_model": vlm_caption_model,
            "target_chars": target_chars,
            "overlap_chars": overlap_chars,
            "chunker": "section" if use_section_chunker else ("docling_semantic" if use_docling_chunker else "baseline"),
            "section_tags": section_tag_keywords,
            "images_saved": "figures_only",
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

    query = "weight layer ?"
    hits = indexer.search(query, top_k=top_k, alpha=alpha, use_neo4j_vector=bool(neo4j_store and cached_chunks is None))

    print("\n[Top results]")
    for ch, combo, ss, bs in hits:
        print(f"- Score={combo:.3f} (sem={ss:.3f} bm25={bs:.3f}) | {ch.source} p.{ch.page} | heading={ch.heading or 'N/A'}")
        print("  ", (ch.text[:180].replace("\n", " ") + "...") if len(ch.text) > 180 else ch.text)
        if getattr(ch, "image_path", None):
            print(f"  image: {ch.image_path}")
        print()

    prompt = build_prompt_with_citations(query, hits)
    print("\n[LLM Answer]")
    _ = groq_stream_completion(prompt)

if __name__ == "__main__":
    main()
