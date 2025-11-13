from dataclasses import dataclass, field
from typing import Optional, Literal, List, Dict, Any
import numpy as np

@dataclass
class Entity:

    name: str  
    canonical_form: str  
    entity_type: str  
    confidence: float  
    chunk_idx: Optional[int] = None  
    position: Optional[int] = None  
    context: Optional[str] = None  
    description: Optional[str] = None  
    embedding: Optional[np.ndarray] = None  
    metadata: Dict[str, Any] = field(default_factory=dict)  

    def __hash__(self):
        return hash((self.canonical_form, self.entity_type))

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.canonical_form == other.canonical_form and self.entity_type == other.entity_type

@dataclass
class Relation:

    subject: Entity  
    predicate: str  
    object: Entity  
    confidence: float  
    chunk_idx: Optional[int] = None  
    evidence: Optional[str] = None  
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash((self.subject, self.predicate, self.object))

    def __eq__(self, other):
        if not isinstance(other, Relation):
            return False
        return (self.subject == other.subject and
                self.predicate == other.predicate and
                self.object == other.object)

@dataclass
class Chunk:
    text: str
    page: int
    source: str
    heading: Optional[str] = None
    modality: Literal["text", "image"] = "text"
    image_path: Optional[str] = None  
    tags: Optional[List[str]] = None
    entities: Optional[List[Entity]] = None  
    relations: Optional[List[Relation]] = None  
