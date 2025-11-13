

from typing import List, Tuple, Optional, Dict, Any
from rag.types import Chunk, Entity

def build_entity_context_section(entities: List[Entity], max_entities: int = 10) -> str:

    if not entities:
        return ""

    sorted_entities = sorted(entities, key=lambda e: e.confidence, reverse=True)
    top_entities = sorted_entities[:max_entities]

    lines = ["### Key Entities\n"]

    for entity in top_entities:
        line = f"- **{entity.canonical_form}** ({entity.entity_type})"

        if entity.description:
            line += f": {entity.description}"

        lines.append(line)

    return "\n".join(lines) + "\n"

def build_relationship_section(subgraph: Dict[str, Any]) -> str:

    if not subgraph or not subgraph.get("edges"):
        return ""

    edges = subgraph.get("edges", [])
    if not edges:
        return ""

    lines = ["### Entity Relationships\n"]

    rel_groups: Dict[str, List[Tuple[str, str]]] = {}

    for edge in edges[:15]:  
        rel_type = edge.get("type", "RELATES_TO")
        source = edge.get("source", "")
        target = edge.get("target", "")

        if rel_type not in rel_groups:
            rel_groups[rel_type] = []

        rel_groups[rel_type].append((source, target))

    for rel_type, pairs in rel_groups.items():
        lines.append(f"\n**{rel_type}**:")
        for source, target in pairs[:5]:  
            lines.append(f"  - {source} → {target}")

    return "\n".join(lines) + "\n"

def format_kg_visualization(subgraph: Dict[str, Any], compact: bool = True) -> str:

    if not subgraph:
        return ""

    nodes = subgraph.get("nodes", [])
    edges = subgraph.get("edges", [])

    if not nodes and not edges:
        return ""

    if compact:

        lines = []
        for edge in edges[:10]:
            source = edge.get("source", "")
            target = edge.get("target", "")
            rel_type = edge.get("type", "→")
            lines.append(f"{source} --[{rel_type}]-> {target}")

        return "\n".join(lines)
    else:

        lines = ["Nodes:"]
        for node in nodes[:10]:
            name = node.get("canonical_form", node.get("name", ""))
            node_type = node.get("type", "")
            lines.append(f"  - {name} ({node_type})")

        lines.append("\nConnections:")
        for edge in edges[:15]:
            source = edge.get("source", "")
            target = edge.get("target", "")
            rel_type = edge.get("type", "→")
            confidence = edge.get("confidence", 0.0)
            lines.append(f"  - {source} --[{rel_type}, {confidence:.2f}]-> {target}")

        return "\n".join(lines)

def build_prompt_with_kg_context(
    query: str,
    hits: List[Tuple[Chunk, float, float, float]],
    entity_context: Optional[Dict[str, Any]] = None,
    include_relationships: bool = True,
    include_graph_viz: bool = False,
) -> str:

    header = (
        "You are a retrieval QA system with access to a knowledge graph. "
        "Use ONLY the provided contexts, entities, and relationships to answer.\n"
        "For every factual statement, add an inline citation in the form "
        "[file:{source}, page:{page}, heading:{heading}].\n"
        "When referencing entities, use the format [entity:{entity_name}].\n"
        "If something is not in the contexts, say you don't know.\n\n"
    )

    sections = []

    if entity_context:
        entities = entity_context.get("entities", [])
        if entities:
            entity_section = build_entity_context_section(entities)
            sections.append(entity_section)

        if include_relationships:
            subgraph = entity_context.get("subgraph", {})
            if subgraph:
                rel_section = build_relationship_section(subgraph)
                if rel_section:
                    sections.append(rel_section)

                if include_graph_viz:
                    graph_viz = format_kg_visualization(subgraph, compact=True)
                    if graph_viz:
                        sections.append("### Knowledge Graph\n```\n" + graph_viz + "\n```\n")

    ctx_lines = []
    for i, (chunk, combo, sem, bm) in enumerate(hits, 1):
        heading = chunk.heading if chunk.heading else "N/A"
        block = (
            f"### Context {i}\n"
            f"Source: file={chunk.source}, page={chunk.page}, heading={heading}\n"
        )

        if chunk.entities:
            entity_names = [e.canonical_form for e in chunk.entities[:5]]
            block += f"Entities: {', '.join(entity_names)}\n"

        if getattr(chunk, "image_path", None):
            block += f"Image: {chunk.image_path}\n"

        block += f"{chunk.text}\n"
        ctx_lines.append(block)

    contexts_section = "\n".join(ctx_lines)
    sections.append(contexts_section)

    prompt_body = "\n".join(sections)
    question_section = f"\n### Question\n{query}\n\n### Answer (with citations):\n"

    return header + prompt_body + question_section

def build_entity_focused_prompt(
    query: str,
    entities: List[Entity],
    entity_chunks: Dict[str, List[Chunk]],
    relationships: Optional[List[Dict]] = None,
) -> str:

    header = (
        "You are answering a question about specific entities using information from documents.\n"
        "Provide accurate information with citations.\n\n"
    )

    sections = []

    if entities:
        lines = ["### Entities in Question\n"]
        for entity in entities:
            line = f"**{entity.canonical_form}** ({entity.entity_type})"
            if entity.description:
                line += f": {entity.description}"
            lines.append(line)

        sections.append("\n".join(lines) + "\n")

    if relationships:
        lines = ["### Known Relationships\n"]
        for rel in relationships[:10]:
            subj = rel.get("subject", "")
            pred = rel.get("predicate", "relates to")
            obj = rel.get("object", "")
            lines.append(f"- {subj} {pred} {obj}")

        sections.append("\n".join(lines) + "\n")

    lines = ["### Relevant Information\n"]
    for entity_name, chunks in entity_chunks.items():
        lines.append(f"\n**About {entity_name}:**\n")

        for i, chunk in enumerate(chunks[:3], 1):
            heading = chunk.heading or "N/A"
            lines.append(
                f"{i}. [{chunk.source}, p.{chunk.page}]: {chunk.text[:200]}..."
            )

    sections.append("\n".join(lines) + "\n")

    question_section = f"### Question\n{query}\n\n### Answer:\n"

    return header + "\n".join(sections) + question_section

def build_prompt_with_citations(
    query: str,
    hits: List[Tuple[Chunk, float, float, float]]
) -> str:

    return build_prompt_with_kg_context(
        query=query,
        hits=hits,
        entity_context=None,
        include_relationships=False,
        include_graph_viz=False,
    )
