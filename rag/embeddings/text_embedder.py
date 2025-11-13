from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(self, model_name: str = "Alibaba-NLP/gte-multilingual-base"):

        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def encode(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])[0]
