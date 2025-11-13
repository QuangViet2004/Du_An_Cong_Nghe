import numpy as np
from typing import List, Optional, Tuple, Dict
from rank_bm25 import BM25Okapi
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from rag.types import Chunk
from rag.embeddings.text_embedder import TextEmbedder
from rag.embeddings.image_captioner import ImageCaptioner

class HybridIndexer:
    def __init__(self,
                 text_model_name: str = "Alibaba-NLP/gte-multilingual-base",
                 image_caption_model: Optional[str] = "nlpconnect/vit-gpt2-image-captioning",
                 use_vlm_caption: bool = True,
                 neo4j_store: Optional[object] = None):
        self.text_embedder = TextEmbedder(text_model_name)
        self.captioner = ImageCaptioner(image_caption_model) if (use_vlm_caption and image_caption_model) else None
        self.chunks: List[Chunk] = []
        self.emb_matrix = None  
        self.bm25: Optional[BM25Okapi] = None
        self._bm25_texts: List[List[str]] = []
        self._lemmatizer = WordNetLemmatizer()
        self._neo4j = neo4j_store

    def build(self, chunks: List[Chunk]) -> None:
        if not chunks:
            raise ValueError("No chunks provided to index.")
        self.chunks = chunks
        texts = [c.text if c.modality == "text" else "" for c in chunks]
        n_text = sum(1 for c in chunks if c.modality == "text")
        n_image = sum(1 for c in chunks if c.modality == "image")
        print(f"[Embed] Encoding text chunks: {n_text}")
        text_emb = self.text_embedder.encode(texts)

        dim = int(text_emb.shape[1])
        vecs = np.zeros((len(chunks), dim), dtype=np.float32)
        captioned = 0
        for i, c in enumerate(chunks):
            if c.modality == "text":
                vecs[i] = text_emb[i]
            elif c.modality == "image" and c.image_path:
                cap = ""
                if self.captioner is not None:
                    try:
                        cap = self.captioner.caption_paths([c.image_path])[0]
                    except Exception:
                        cap = ""

                if cap:
                    c.text = cap
                    vecs[i] = self.text_embedder.encode([cap])[0].astype(np.float32)
                    captioned += 1
                else:
                    vecs[i] = np.zeros((dim,), dtype=np.float32)
            else:
                vecs[i] = np.zeros((dim,), dtype=np.float32)
        self.emb_matrix = vecs
        if n_image:
            print(f"[Caption] Generated captions for {captioned}/{n_image} images")
        print(f"[Embed] Embedding matrix shape: {self.emb_matrix.shape}, dtype: {self.emb_matrix.dtype}")

        bm25_texts = [c.text if c.text else "" for c in self.chunks]
        self._bm25_texts = [t.lower().split() for t in bm25_texts]
        try:
            self.bm25 = BM25Okapi(self._bm25_texts) if len(self._bm25_texts) > 0 else None
        except Exception:

            self.bm25 = None
        print(f"[BM25] Initialized over {len(self._bm25_texts)} documents")

        if self._neo4j is not None and self.emb_matrix is not None:
            try:
                self._neo4j.ensure_indexes(dim=int(self.emb_matrix.shape[1]))
                rows = []
                for idx, ch in enumerate(self.chunks):
                    rows.append({
                        "source": ch.source,
                        "page": ch.page,
                        "heading": ch.heading or "",
                        "idx": idx,
                        "text": ch.text,
                        "image_path": getattr(ch, "image_path", None),
                        "embedding": self.emb_matrix[idx].tolist(),
                    })
                self._neo4j.upsert_rows(rows)
                print(f"[Neo4j] Upserted {len(rows)} chunk nodes (with vectors)")
            except Exception:
                pass

    def build_with_embeddings(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        if not chunks or embeddings is None or len(chunks) != embeddings.shape[0]:
            raise ValueError("Chunks and embeddings must be provided with matching lengths.")
        self.chunks = chunks
        self.emb_matrix = embeddings.astype(np.float32)

        texts = [c.text if c.modality == "text" else "" for c in chunks]
        self._bm25_texts = [t.lower().split() for t in texts]
        try:
            self.bm25 = BM25Okapi(self._bm25_texts) if len(self._bm25_texts) > 0 else None
        except Exception:
            self.bm25 = None

    def _expand_query(self, q: str) -> str:
        toks = q.split()
        bag = set(toks)
        for w in toks:
            lw = self._lemmatizer.lemmatize(w.lower())
            bag.add(lw)
            for syn in wordnet.synsets(w):
                for lemma in syn.lemmas():
                    bag.add(lemma.name().lower().replace("_", " "))
        return " ".join(bag)

    def search(self, query: str, top_k: int = 5, alpha: float = 0.6, use_neo4j_vector: bool = False) -> List[Tuple[Chunk, float, float, float]]:
        if self.emb_matrix is None:
            raise RuntimeError("Indexer not built. Call build(chunks) first.")

        q_emb = self.text_embedder.encode_query(query).astype(np.float32)
        sem_scores = self.emb_matrix @ q_emb

        if use_neo4j_vector and self._neo4j is not None:
            try:
                neo_hits = self._neo4j.vector_search(q_emb, top_k=top_k * 3)

                for props, score in neo_hits:
                    src = props.get("source")
                    page = int(props.get("page", 0))
                    idx = int(props.get("idx", -1))
                    if 0 <= idx < len(self.chunks):
                        ch = self.chunks[idx]
                        if ch.source == src and ch.page == page:
                            sem_scores[idx] = max(sem_scores[idx], float(score))
            except Exception:
                pass

        expanded = self._expand_query(query)
        if self.bm25 is not None:
            bm25_scores = self.bm25.get_scores(expanded.lower().split())
        else:
            bm25_scores = np.zeros(len(self.chunks), dtype=np.float32)

        def _norm(x):
            x = np.asarray(x, dtype=np.float32)
            mn, mx = float(x.min()), float(x.max())
            if mx - mn < 1e-8:
                return np.zeros_like(x)
            return (x - mn) / (mx - mn)

        sem_n = _norm(sem_scores)
        bm25_n = _norm(bm25_scores)
        combined = alpha * sem_n + (1 - alpha) * bm25_n

        idxs = np.argsort(-combined)[:top_k]
        results = []
        for i in idxs:
            results.append((self.chunks[i], float(combined[i]), float(sem_scores[i]), float(bm25_scores[i])))
        return results
