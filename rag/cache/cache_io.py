import hashlib
import json
import os
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from rag.types import Chunk

def compute_dataset_md5(files: List[str]) -> str:
    h = hashlib.md5()
    for fp in files:
        if not os.path.isfile(fp):

            h.update(fp.encode("utf-8"))
            continue
        h.update(os.path.basename(fp).encode("utf-8"))
        h.update(str(os.path.getsize(fp)).encode("utf-8"))
        with open(fp, "rb") as f:
            while True:
                b = f.read(1024 * 1024)
                if not b:
                    break
                h.update(b)
    return h.hexdigest()

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _chunks_to_dicts(chunks: List[Chunk]) -> List[Dict[str, Any]]:
    out = []
    for c in chunks:
        out.append({
            "text": c.text,
            "page": c.page,
            "source": c.source,
            "heading": c.heading,
            "modality": getattr(c, "modality", "text"),
            "image_path": getattr(c, "image_path", None),
            "tags": getattr(c, "tags", None),
        })
    return out

def _dicts_to_chunks(items: List[Dict[str, Any]]) -> List[Chunk]:
    out: List[Chunk] = []
    for d in items:
        out.append(Chunk(
            text=d.get("text", ""),
            page=int(d.get("page", 0)),
            source=d.get("source", ""),
            heading=d.get("heading"),
            modality=d.get("modality", "text"),
            image_path=d.get("image_path"),
            tags=d.get("tags"),
        ))
    return out

def cache_paths(cache_dir: str, key: str) -> Dict[str, str]:
    base = os.path.join(cache_dir, key)
    return {
        "dir": base,
        "chunks": os.path.join(base, "chunks.json"),
        "emb": os.path.join(base, "embeddings.npy"),
        "meta": os.path.join(base, "meta.json"),
        "graph": os.path.join(base, "graph.json"),
    }

def save_cache(cache_dir: str,
               key: str,
               chunks: List[Chunk],
               emb: np.ndarray,
               meta: Dict[str, Any],
               graph_rows: Optional[List[Dict[str, Any]]] = None) -> None:
    paths = cache_paths(cache_dir, key)
    _ensure_dir(paths["dir"])
    with open(paths["chunks"], "w", encoding="utf-8") as f:
        json.dump(_chunks_to_dicts(chunks), f, ensure_ascii=False)
    np.save(paths["emb"], emb)
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    if graph_rows is not None:
        with open(paths["graph"], "w", encoding="utf-8") as f:
            json.dump(graph_rows, f, ensure_ascii=False)

def load_cache(cache_dir: str, key: str) -> Tuple[Optional[List[Chunk]], Optional[np.ndarray], Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]]:
    paths = cache_paths(cache_dir, key)
    if not (os.path.exists(paths["chunks"]) and os.path.exists(paths["emb"]) and os.path.exists(paths["meta"])):
        return None, None, None, None
    try:
        with open(paths["chunks"], "r", encoding="utf-8") as f:
            chunks = _dicts_to_chunks(json.load(f))
        emb = np.load(paths["emb"])  
        with open(paths["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)
        graph_rows = None
        if os.path.exists(paths["graph"]):
            with open(paths["graph"], "r", encoding="utf-8") as f:
                graph_rows = json.load(f)
        return chunks, emb, meta, graph_rows
    except Exception:
        return None, None, None, None
