

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class KGExtractionConfig:

    mode: str = "hybrid"

    entity_types: List[str] = field(default_factory=lambda: [
        "Person", "Organization", "Location", "Concept",
        "Date", "Number", "Technical_Term", "Product", "Event"
    ])

    min_confidence: float = 0.7

    enable_coreference: bool = False

    spacy_model: str = "en_core_web_sm"

    llm_model: str = "llama-3.3-70b-versatile"

    enable_llm_refinement: bool = True

@dataclass
class KGRetrievalConfig:

    retrieval_mode: str = "hybrid"

    expansion_hops: int = 2

    entity_boost_weight: float = 0.3

    centrality_weight: float = 0.2

    include_entity_context: bool = True

@dataclass
class KGPipelineConfig:

    enabled: bool = False

    extraction: KGExtractionConfig = field(default_factory=KGExtractionConfig)

    retrieval: KGRetrievalConfig = field(default_factory=KGRetrievalConfig)

    enable_enrichment: bool = True

    store_in_neo4j: bool = True

FAST_MODE = KGPipelineConfig(
    enabled=True,
    extraction=KGExtractionConfig(
        mode="ner_only",
        min_confidence=0.8,
        enable_coreference=False,
        enable_llm_refinement=False,
    ),
    retrieval=KGRetrievalConfig(
        retrieval_mode="entity_aware",
        expansion_hops=1,
        entity_boost_weight=0.2,
        centrality_weight=0.1,
        include_entity_context=False,
    ),
    enable_enrichment=False,
)

BALANCED_MODE = KGPipelineConfig(
    enabled=True,
    extraction=KGExtractionConfig(
        mode="hybrid",
        min_confidence=0.7,
        enable_coreference=False,
        enable_llm_refinement=True,
    ),
    retrieval=KGRetrievalConfig(
        retrieval_mode="hybrid",
        expansion_hops=2,
        entity_boost_weight=0.3,
        centrality_weight=0.2,
        include_entity_context=True,
    ),
    enable_enrichment=True,
)

COMPREHENSIVE_MODE = KGPipelineConfig(
    enabled=True,
    extraction=KGExtractionConfig(
        mode="hybrid",
        min_confidence=0.6,
        enable_coreference=True,
        enable_llm_refinement=True,
    ),
    retrieval=KGRetrievalConfig(
        retrieval_mode="hybrid",
        expansion_hops=3,
        entity_boost_weight=0.4,
        centrality_weight=0.3,
        include_entity_context=True,
    ),
    enable_enrichment=True,
)

PRESET_MODES = {
    "fast": FAST_MODE,
    "balanced": BALANCED_MODE,
    "comprehensive": COMPREHENSIVE_MODE,
}

def get_kg_config(mode: str = "balanced") -> KGPipelineConfig:

    if mode in PRESET_MODES:
        return PRESET_MODES[mode]
    else:
        print(f"[KGConfig] Unknown mode '{mode}', using 'balanced'")
        return BALANCED_MODE

def create_custom_config(
    enabled: bool = True,
    extraction_mode: str = "hybrid",
    retrieval_mode: str = "hybrid",
    **kwargs
) -> KGPipelineConfig:

    extraction_config = KGExtractionConfig(
        mode=extraction_mode,
        **{k: v for k, v in kwargs.items() if k in KGExtractionConfig.__dataclass_fields__}
    )

    retrieval_config = KGRetrievalConfig(
        retrieval_mode=retrieval_mode,
        **{k: v for k, v in kwargs.items() if k in KGRetrievalConfig.__dataclass_fields__}
    )

    pipeline_config = KGPipelineConfig(
        enabled=enabled,
        extraction=extraction_config,
        retrieval=retrieval_config,
        **{k: v for k, v in kwargs.items()
           if k in KGPipelineConfig.__dataclass_fields__
           and k not in ['extraction', 'retrieval']}
    )

    return pipeline_config
