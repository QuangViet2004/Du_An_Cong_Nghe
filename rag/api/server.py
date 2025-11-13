from typing import List, Optional, Dict, Any
import os

from fastapi import FastAPI, HTTPException
from fastapi import UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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
from rag_demo import build_prompt_with_citations

DEFAULT_TEXT_MODEL = "Alibaba-NLP/gte-multilingual-base"
DEFAULT_VLM_MODEL = "ibm-granite/granite-docling-258M"

class IndexRequest(BaseModel):
    pdf_files: List[str]
    use_section_chunker: bool = True
    section_tag_keywords: List[str] = Field(default_factory=list)
    use_docling_chunker: bool = True
    target_chars: int = 900
    overlap_chars: int = 150
    text_model: str = DEFAULT_TEXT_MODEL
    vlm_caption_model: str = DEFAULT_VLM_MODEL
    use_neo4j: bool = True
    neo4j_uri: str = Field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = Field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = Field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password"))

class IndexResponse(BaseModel):
    dataset_key: str
    cached: bool
    n_chunks: int
    n_text: int
    n_image: int
    meta: Dict[str, Any]

class SearchResponseItem(BaseModel):
    score: float
    sem: float
    bm25: float
    text: str
    page: int
    source: str
    heading: Optional[str]
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    tags: Optional[List[str]] = None

class SearchResponse(BaseModel):
    dataset_key: str
    query: str
    top_k: int
    alpha: float
    hits: List[SearchResponseItem]

class PromptResponse(BaseModel):
    dataset_key: str
    query: str
    top_k: int
    alpha: float
    prompt: str

class AskResponse(BaseModel):
    dataset_key: str
    query: str
    top_k: int
    alpha: float
    message: str

app = FastAPI(title="RAG API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("CORS_ALLOW_ORIGINS", "*")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_CACHE_DIR = os.path.join(os.getcwd(), ".rag_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
app.mount("/cache", StaticFiles(directory=_CACHE_DIR), name="cache")

@app.post("/index", response_model=IndexResponse)
def index_docs(req: IndexRequest):
    if not req.pdf_files:
        raise HTTPException(status_code=400, detail="pdf_files must not be empty")

    dataset_key = compute_dataset_md5(req.pdf_files)
    cache_dir = os.path.join(os.getcwd(), ".rag_cache")
    image_output_dir = os.path.join(cache_dir, dataset_key, "images")
    os.makedirs(image_output_dir, exist_ok=True)

    cached_chunks, cached_emb, cached_meta, cached_graph = load_cache(cache_dir, dataset_key)
    if cached_meta is not None:
        desired_chunker = "section" if req.use_section_chunker else ("docling_semantic" if req.use_docling_chunker else "baseline")
        if (
            cached_meta.get("text_model") != req.text_model
            or cached_meta.get("image_caption_model") != req.vlm_caption_model
            or cached_meta.get("chunker") != desired_chunker
            or cached_meta.get("images_saved") != "figures_only"
        ):
            cached_chunks, cached_emb = None, None

    neo4j_store: Optional[Neo4jGraphStore] = None
    if req.use_neo4j:
        neo4j_store = Neo4jGraphStore(req.neo4j_uri, req.neo4j_user, req.neo4j_password)

    indexer = HybridIndexer(
        text_model_name=req.text_model,
        image_caption_model=req.vlm_caption_model,
        use_vlm_caption=True,
        neo4j_store=neo4j_store,
    )

    if cached_chunks is not None and cached_emb is not None:
        indexer.build_with_embeddings(cached_chunks, cached_emb)
        fine_chunks = cached_chunks
        n_text = sum(1 for c in fine_chunks if c.modality == "text" and c.text)
        n_image = sum(1 for c in fine_chunks if c.modality == "image")
        meta = cached_meta or {}
        return IndexResponse(
            dataset_key=dataset_key,
            cached=True,
            n_chunks=len(fine_chunks),
            n_text=n_text,
            n_image=n_image,
            meta=meta,
        )

    raw_chunks: List[Chunk] = []
    for pdf in req.pdf_files:
        raw_chunks.extend(load_pdf_with_docling(pdf, image_out_dir=image_output_dir))
    n_text_raw = sum(1 for c in raw_chunks if c.modality == "text" and c.text)
    n_img_raw = sum(1 for c in raw_chunks if c.modality == "image")

    text_embedder = TextEmbedder(model_name=req.text_model)
    if req.use_section_chunker:
        fine_chunks = section_chunk_blocks(raw_chunks, tag_keywords=req.section_tag_keywords)
    elif req.use_docling_chunker:
        fine_chunks = semantic_chunk_blocks_docling(
            raw_chunks,
            encode_sentences=lambda sents: text_embedder.encode(sents),
            target_chars=req.target_chars,
            overlap_chars=req.overlap_chars,
        )
    else:
        fine_chunks = semantic_chunk_blocks(raw_chunks, target_chars=req.target_chars, overlap_chars=req.overlap_chars)

    indexer.build(fine_chunks)

    meta = {
        "text_model": req.text_model,
        "image_caption_model": req.vlm_caption_model,
        "target_chars": req.target_chars,
        "overlap_chars": req.overlap_chars,
        "chunker": "section" if req.use_section_chunker else ("docling_semantic" if req.use_docling_chunker else "baseline"),
        "section_tags": req.section_tag_keywords,
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

    n_text = sum(1 for c in fine_chunks if c.modality == "text" and c.text)
    n_image = sum(1 for c in fine_chunks if c.modality == "image")
    return IndexResponse(
        dataset_key=dataset_key,
        cached=False,
        n_chunks=len(fine_chunks),
        n_text=n_text,
        n_image=n_image,
        meta=meta,
    )

@app.post("/upload-index", response_model=IndexResponse)
async def upload_index(
    files: List[UploadFile] = File(...),
    use_section_chunker: bool = Form(True),
    section_tag_keywords: str = Form(""),
    use_docling_chunker: bool = Form(True),
    target_chars: int = Form(900),
    overlap_chars: int = Form(150),
    text_model: str = Form(DEFAULT_TEXT_MODEL),
    vlm_caption_model: str = Form(DEFAULT_VLM_MODEL),
    use_neo4j: bool = Form(True),
    neo4j_uri: str = Form(os.getenv("NEO4J_URI", "bolt://localhost:7687")),
    neo4j_user: str = Form(os.getenv("NEO4J_USER", "neo4j")),
    neo4j_password: str = Form(os.getenv("NEO4J_PASSWORD", "password")),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    uploads_root = os.path.join(_CACHE_DIR, "uploads")
    os.makedirs(uploads_root, exist_ok=True)
    pid_dir = os.path.join(uploads_root, str(os.getpid()))
    os.makedirs(pid_dir, exist_ok=True)

    saved_paths: List[str] = []
    for uf in files:
        if not uf.filename or not uf.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {uf.filename}")
        dst = os.path.join(pid_dir, uf.filename)
        try:
            data = await uf.read()
            with open(dst, "wb") as f:
                f.write(data)
            saved_paths.append(dst)
        finally:
            await uf.close()

    tags_list = [t.strip() for t in section_tag_keywords.split(",") if t.strip()] if section_tag_keywords else []

    dataset_key = compute_dataset_md5(saved_paths)
    cache_dir = _CACHE_DIR
    image_output_dir = os.path.join(cache_dir, dataset_key, "images")
    os.makedirs(image_output_dir, exist_ok=True)

    cached_chunks, cached_emb, cached_meta, cached_graph = load_cache(cache_dir, dataset_key)
    if cached_meta is not None:
        desired_chunker = "section" if use_section_chunker else ("docling_semantic" if use_docling_chunker else "baseline")
        if (
            cached_meta.get("text_model") != text_model
            or cached_meta.get("image_caption_model") != vlm_caption_model
            or cached_meta.get("chunker") != desired_chunker
            or cached_meta.get("images_saved") != "figures_only"
        ):
            cached_chunks, cached_emb = None, None

    neo4j_store: Optional[Neo4jGraphStore] = None
    if use_neo4j:
        neo4j_store = Neo4jGraphStore(neo4j_uri, neo4j_user, neo4j_password)

    indexer = HybridIndexer(
        text_model_name=text_model,
        image_caption_model=vlm_caption_model,
        use_vlm_caption=True,
        neo4j_store=neo4j_store,
    )

    if cached_chunks is not None and cached_emb is not None:
        indexer.build_with_embeddings(cached_chunks, cached_emb)
        fine_chunks = cached_chunks
        n_text = sum(1 for c in fine_chunks if c.modality == "text" and c.text)
        n_image = sum(1 for c in fine_chunks if c.modality == "image")
        meta = cached_meta or {}
        return IndexResponse(
            dataset_key=dataset_key,
            cached=True,
            n_chunks=len(fine_chunks),
            n_text=n_text,
            n_image=n_image,
            meta=meta,
        )

    raw_chunks: List[Chunk] = []
    for pdf in saved_paths:
        raw_chunks.extend(load_pdf_with_docling(pdf, image_out_dir=image_output_dir))

    text_embedder = TextEmbedder(model_name=text_model)
    if use_section_chunker:
        fine_chunks = section_chunk_blocks(raw_chunks, tag_keywords=tags_list)
    elif use_docling_chunker:
        fine_chunks = semantic_chunk_blocks_docling(
            raw_chunks,
            encode_sentences=lambda sents: text_embedder.encode(sents),
            target_chars=target_chars,
            overlap_chars=overlap_chars,
        )
    else:
        fine_chunks = semantic_chunk_blocks(raw_chunks, target_chars=target_chars, overlap_chars=overlap_chars)

    indexer.build(fine_chunks)

    meta = {
        "text_model": text_model,
        "image_caption_model": vlm_caption_model,
        "target_chars": target_chars,
        "overlap_chars": overlap_chars,
        "chunker": "section" if use_section_chunker else ("docling_semantic" if use_docling_chunker else "baseline"),
        "section_tags": tags_list,
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

    n_text = sum(1 for c in fine_chunks if c.modality == "text" and c.text)
    n_image = sum(1 for c in fine_chunks if c.modality == "image")
    return IndexResponse(
        dataset_key=dataset_key,
        cached=False,
        n_chunks=len(fine_chunks),
        n_text=n_text,
        n_image=n_image,
        meta=meta,
    )

@app.get("/search", response_model=SearchResponse)
def search(dataset_key: str, q: str, top_k: int = 5, alpha: float = 0.6,
           use_neo4j_vector: Optional[bool] = None,
           neo4j_uri: Optional[str] = None,
           neo4j_user: Optional[str] = None,
           neo4j_password: Optional[str] = None):
    cache_dir = os.path.join(os.getcwd(), ".rag_cache")
    chunks, emb, meta, graph_rows = load_cache(cache_dir, dataset_key)
    if chunks is None or emb is None or meta is None:
        raise HTTPException(status_code=404, detail="Dataset not indexed or cache missing")

    neo4j_store = None
    if use_neo4j_vector is None:
        use_neo4j_vector = True
    if use_neo4j_vector:
        uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        pwd = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
        neo4j_store = Neo4jGraphStore(uri, user, pwd)

    text_model = meta.get("text_model", DEFAULT_TEXT_MODEL)
    vlm_model = meta.get("image_caption_model", DEFAULT_VLM_MODEL)
    indexer = HybridIndexer(text_model_name=text_model, image_caption_model=vlm_model, use_vlm_caption=True, neo4j_store=neo4j_store)
    indexer.build_with_embeddings(chunks, emb)

    hits = indexer.search(q, top_k=top_k, alpha=alpha, use_neo4j_vector=bool(neo4j_store))
    items: List[SearchResponseItem] = []
    for ch, combo, ss, bs in hits:
        img_url: Optional[str] = None
        ip = getattr(ch, "image_path", None)
        if ip and os.path.abspath(ip).startswith(os.path.abspath(_CACHE_DIR)):
            rel = os.path.relpath(ip, _CACHE_DIR).replace("\\", "/")
            img_url = f"/cache/{rel}"
        items.append(SearchResponseItem(
            score=combo,
            sem=ss,
            bm25=bs,
            text=ch.text,
            page=ch.page,
            source=ch.source,
            heading=ch.heading,
            image_path=getattr(ch, "image_path", None),
            image_url=img_url,
            tags=getattr(ch, "tags", None),
        ))
    return SearchResponse(dataset_key=dataset_key, query=q, top_k=top_k, alpha=alpha, hits=items)

@app.get("/prompt", response_model=PromptResponse)
def get_prompt(dataset_key: str, q: str, top_k: int = 5, alpha: float = 0.6):
    cache_dir = os.path.join(os.getcwd(), ".rag_cache")
    chunks, emb, meta, graph_rows = load_cache(cache_dir, dataset_key)
    if chunks is None or emb is None or meta is None:
        raise HTTPException(status_code=404, detail="Dataset not indexed or cache missing")

    indexer = HybridIndexer(text_model_name=meta.get("text_model", DEFAULT_TEXT_MODEL), image_caption_model=meta.get("image_caption_model", DEFAULT_VLM_MODEL), use_vlm_caption=True)
    indexer.build_with_embeddings(chunks, emb)
    hits = indexer.search(q, top_k=top_k, alpha=alpha)
    prompt = build_prompt_with_citations(q, hits)
    return PromptResponse(dataset_key=dataset_key, query=q, top_k=top_k, alpha=alpha, prompt=prompt)

@app.post("/ask", response_model=AskResponse)
def ask_llm(dataset_key: str, q: str, top_k: int = 5, alpha: float = 0.6):
    cache_dir = os.path.join(os.getcwd(), ".rag_cache")
    chunks, emb, meta, graph_rows = load_cache(cache_dir, dataset_key)
    if chunks is None or emb is None or meta is None:
        raise HTTPException(status_code=404, detail="Dataset not indexed or cache missing")

    indexer = HybridIndexer(text_model_name=meta.get("text_model", DEFAULT_TEXT_MODEL), image_caption_model=meta.get("image_caption_model", DEFAULT_VLM_MODEL), use_vlm_caption=True)
    indexer.build_with_embeddings(chunks, emb)
    hits = indexer.search(q, top_k=top_k, alpha=alpha)
    prompt = build_prompt_with_citations(q, hits)
    groq_stream_completion(prompt)
    return AskResponse(dataset_key=dataset_key, query=q, top_k=top_k, alpha=alpha, message="Streaming started on server logs.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("rag.api.server:app", host="0.0.0.0", port=8000, reload=False)
