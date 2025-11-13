import os
import sys
from typing import List, Tuple

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if os.name == "nt" and not os.getenv("HF_HUB_DISABLE_SYMLINKS"):
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from rag.loaders.doc_loader import load_pdf_with_docling
from rag.chunking.semantic_chunker import semantic_chunk_blocks
from rag.retrieval.hybrid_indexer import HybridIndexer
from rag.prompts.citation_prompt import build_prompt_with_citations
from rag.llm.groq_client import groq_stream_completion
from rag.graph.kg import KnowledgeGraph
from rag.graph.graph_flow import build_kg_flow
from rag.types import Chunk

def main():
    pdf_files = [
        r"1512.03385v1.pdf",
    ]
    target_chars = 900
    overlap_chars = 150
    top_k = 5
    alpha = 0.6

    raw_chunks: List[Chunk] = []
    for pdf in pdf_files:
        raw_chunks.extend(load_pdf_with_docling(pdf))

    print('raw_chunks ---> ', raw_chunks)
    fine_chunks = semantic_chunk_blocks(raw_chunks, target_chars=target_chars, overlap_chars=overlap_chars)

    print('fine_chunks ---> ', fine_chunks)

    kg = KnowledgeGraph()
    flow = build_kg_flow(kg)
    flow.invoke({"texts": [c.text for c in fine_chunks if c.modality == "text"]})

    indexer = HybridIndexer(text_model_name="Alibaba-NLP/gte-multilingual-base")
    indexer.build(fine_chunks)

    query = "độ chính xác của phương pháp này như nào?"
    hits = indexer.search(query, top_k=top_k, alpha=alpha)

    print("\n[Top results]")
    for ch, combo, ss, bs in hits:
        preview = (ch.text[:180].replace("\n", " ") + "...") if ch.text and len(ch.text) > 180 else (ch.text or "[image]")
        print(f"- Score={combo:.3f} (sem={ss:.3f} bm25={bs:.3f}) | {ch.source} p.{ch.page} | heading={ch.heading or 'N/A'}")
        print("  ", preview)
        print()

    prompt = build_prompt_with_citations(query, hits)
    print("\n[LLM Answer]")
    _ = groq_stream_completion(prompt)

if __name__ == "__main__":

    if not os.getenv("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY not set. Set it to call Groq.")
    main()
