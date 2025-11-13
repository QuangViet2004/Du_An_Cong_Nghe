from typing import List
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

class ImageEmbedder:

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def encode_paths(self, image_paths: List[str]) -> np.ndarray:
        images = [Image.open(p).convert("RGB") for p in image_paths]
        emb = self.model.encode(images, convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype(np.float32)
