from typing import List, Tuple
from rag.types import Chunk

def build_prompt_with_citations(query: str, hits: List[Tuple[Chunk, float, float, float]]) -> str:
    header = (
        "You are a retrieval QA system. Use ONLY the provided contexts to answer.\n"
        "For every factual statement, add an inline citation in the form "
        "[file:{source}, page:{page}, heading:{heading}].\n"
        "If something is not in the contexts, say you don't know.\n\n"
    )
    ctx = []
    for i, (chunk, combo, sem, bm) in enumerate(hits, 1):
        heading = chunk.heading if chunk.heading else "N/A"
        content = chunk.text if chunk.modality == "text" else f"[Image at {chunk.image_path}]"
        ctx.append(
            f"### Context {i}\n"
            f"Source: file={chunk.source}, page={chunk.page}, heading={heading}\n"
            f"{content}\n"
        )
    context_block = "\n".join(ctx)
    ask = f"\n### Question\n{query}\n\n### Answer (with citations):\n"
    return header + context_block + ask
